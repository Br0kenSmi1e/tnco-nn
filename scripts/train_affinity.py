"""Train/evaluate the affinity model: predict oracle merge-times.

GPU-only. Uses existing oracle data (data/train_actions.jsonl).
Reports Spearman correlation on held-out instances for:
  - the GNN (AffinityNet)
  - control: shortest-path distance
  - control: |deg_i - deg_j| / product features (simple linear probe)

Usage: .venv/bin/python scripts/train_affinity.py [--epochs 60]
"""

import argparse
import json
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnco.affinity import AffinityNet, lca_sizes, shortest_path_matrix, static_graph
from tnco.instances import Instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PAIRS_PER_INST = 400


def spearman(pred: torch.Tensor, target: torch.Tensor) -> float:
    def rank(x):
        return x.argsort().argsort().float()
    a, b = rank(pred), rank(target)
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-12)


def load_split(seed: int = 0):
    recs = [json.loads(l) for l in open(os.path.join(DATA, "train_actions.jsonl"))]
    rng = random.Random(seed)
    rng.shuffle(recs)
    n_val = max(1, len(recs) // 10)
    return recs[n_val:], recs[:n_val]


def prepare(rec, rni: int = 0):
    inst = Instance.load(os.path.join(DATA, "instances", "train", rec["instance"] + ".json"))
    nf, edges, ef = static_graph(inst)
    oracle = json.load(open(os.path.join(DATA, "oracle", "train", rec["instance"] + ".json")))
    m = lca_sizes(inst, oracle["tree"])
    if rni:
        rng = random.Random(7)
        extra = [[rng.gauss(0, 1) for _ in range(rni)] for _ in range(len(nf))]
        nf = [f + e for f, e in zip(nf, extra)]
    n = m.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng = random.Random(0)
    rng.shuffle(pairs)
    pairs = pairs[:PAIRS_PER_INST]
    tgt = torch.tensor([m[i, j] for i, j in pairs], dtype=torch.float32)
    return {"nf": nf, "edges": edges, "ef": ef, "pairs": pairs, "tgt": tgt,
            "inst": inst, "n": n, "name": rec["instance"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--rni", type=int, default=0,
                    help="number of random node-identifier features to append")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--emb", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    train_recs, val_recs = load_split()
    print(f"{len(train_recs)} train / {len(val_recs)} val instances")
    train = [prepare(r, args.rni) for r in train_recs]
    val = [prepare(r, args.rni) for r in val_recs]

    model = AffinityNet(args.hidden, args.layers, args.emb, in_dim=3 + args.rni).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def to_dev(d):
        return (
            torch.tensor(d["nf"], dtype=torch.float32, device=device),
            torch.tensor(d["edges"], dtype=torch.long, device=device),
            torch.tensor(d["ef"], dtype=torch.float32, device=device),
            torch.tensor(d["pairs"], dtype=torch.long, device=device),
            d["tgt"].to(device),
        )

    train_d = [to_dev(d) for d in train]
    val_d = [to_dev(d) for d in val]

    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        order = list(range(len(train_d)))
        random.shuffle(order)
        for i in order:
            nf, ed, ef, pairs, tgt = train_d[i]
            pred = model(nf, ed, ef, pairs)
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                sp = []
                for nf, ed, ef, pairs, tgt in val_d:
                    pred = model(nf, ed, ef, pairs)
                    sp.append(spearman(pred, tgt))
            print(f"epoch {epoch:3d} train mse {tot / len(train_d):.5f} "
                  f"| val spearman {sum(sp) / len(sp):.3f}", flush=True)

    # controls on val, overall and per family
    from collections import defaultdict
    sp_model_by_fam, sp_dist_by_fam = defaultdict(list), defaultdict(list)
    model.eval()
    with torch.no_grad():
        for d, (nf, ed, ef, pairs, tgt) in zip(val, val_d):
            fam = d["name"].split("_")[0]
            pred = model(nf, ed, ef, pairs)
            sp_model_by_fam[fam].append(spearman(pred, tgt))
            dm = shortest_path_matrix(d["inst"])
            dist_pred = torch.tensor([dm[i, j] for i, j in d["pairs"]], dtype=torch.float32)
            sp_dist_by_fam[fam].append(spearman(dist_pred.to(device), tgt))
    for fam in sorted(sp_model_by_fam):
        gm = sp_model_by_fam[fam]
        cm = sp_dist_by_fam[fam]
        print(f"family {fam:5s} (n={len(gm)}): GNN spearman {sum(gm)/len(gm):.3f} "
              f"| shortest-path {sum(cm)/len(cm):.3f}")

    os.makedirs(os.path.join(DATA, "affinity"), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden": args.hidden,
                "layers": args.layers, "emb": args.emb},
               os.path.join(DATA, "affinity", "affinity.pt"))


if __name__ == "__main__":
    main()

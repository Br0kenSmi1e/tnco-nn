"""Train the EdgePolicy by imitating TreeSA oracle action sequences.

Teacher forcing: at each oracle step, featurize the state, CE over the
contractible-edge logits with the oracle's index as target.

Usage: .venv/bin/python scripts/train_imitation.py [--epochs 30] [--hidden 128] [--layers 6]
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

from tnco.contraction import ContractionState
from tnco.gnn import EdgePolicy, featurize
from tnco.instances import Instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def load_dataset(path: str):
    """Pre-compute all (state, target-edge-position) steps."""
    samples = []  # (node_feat, edges, edge_feat, target_pos)
    for line in open(path):
        rec = json.loads(line)
        inst = Instance.load(os.path.join(DATA, "instances", "train", rec["instance"] + ".json"))
        state = ContractionState.initial(inst)
        for a in rec["actions"]:
            nf, ef, eff, labels = featurize(state)
            try:
                pos = labels.index(a)
            except ValueError:
                break  # action not contractible (shouldn't happen)
            samples.append((nf, ef, eff, pos, rec["instance"]))
            state = state.contract(a)
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(DATA, "policy.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading dataset...")
    samples = load_dataset(os.path.join(DATA, "train_actions.jsonl"))
    # hold out a few instances for validation (by name)
    rng = random.Random(0)
    names = sorted({s[4] for s in samples})
    val_names = set(rng.sample(names, max(1, len(names) // 10)))
    train = [s for s in samples if s[4] not in val_names]
    val = [s for s in samples if s[4] in val_names]
    print(f"{len(train)} train steps, {len(val)} val steps from {len(names)} instances")

    model = EdgePolicy(args.hidden, args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run_epoch(data, train_mode: bool) -> tuple[float, float]:
        order = list(range(len(data)))
        if train_mode:
            rng.shuffle(order)
        total_loss = total_acc = 0.0
        for i in range(0, len(order), args.batch):
            idxs = order[i : i + args.batch]
            loss = acc = 0.0
            if train_mode:
                opt.zero_grad()
            losses = []
            for j in idxs:
                nf, ef, eff, pos, _ = data[j]
                nf_t = torch.tensor(nf, dtype=torch.float32, device=device)
                ef_t = torch.tensor(ef, dtype=torch.long, device=device)
                eff_t = torch.tensor(eff, dtype=torch.float32, device=device)
                logits = model(nf_t, ef_t, eff_t, len(nf))
                losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([pos], device=device)))
                acc += float(logits.argmax().item() == pos)
            bl = torch.stack(losses).mean()
            if train_mode:
                bl.backward()
                opt.step()
            total_loss += bl.item() * len(idxs)
            total_acc += acc
        return total_loss / len(data), total_acc / len(data)

    for epoch in range(args.epochs):
        model.train()
        tl, ta = run_epoch(train, True)
        model.eval()
        with torch.no_grad():
            vl, va = run_epoch(val, False)
        print(f"epoch {epoch:3d} train loss {tl:.4f} acc {ta:.3f} | val loss {vl:.4f} acc {va:.3f}", flush=True)

    torch.save({"state_dict": model.state_dict(), "hidden": args.hidden, "layers": args.layers}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()

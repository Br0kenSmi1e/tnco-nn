"""Expert iteration: fine-tune the imitation policy against the exact
cost oracle by self-imitating the best-of-k rollouts.

Each round:
  - sample B training instances, k rollouts each (1 greedy + k-1 sampled)
  - per instance, pick the cheapest trajectory (exact tc)
  - teacher-forced CE on winning trajectories
  - mix in oracle trajectories (weight w_oracle) to prevent drift

Usage: .venv/bin/python scripts/expert_iteration.py [--rounds 60]
"""

import argparse
import glob
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnco.gnn import EdgePolicy
from tnco.instances import Instance
from tnco.rollout import rollout_batch, trajectory_ce_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def load_train_pool():
    insts = {}
    for p in sorted(glob.glob(os.path.join(DATA, "instances", "train", "*.json"))):
        name = os.path.basename(p)[:-5]
        insts[name] = Instance.load(p, name=name)
    oracle = {}
    for line in open(os.path.join(DATA, "train_actions.jsonl")):
        rec = json.loads(line)
        oracle[rec["instance"]] = (rec["actions"], rec["oracle_tc"])
    return insts, oracle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--batch-instances", type=int, default=64)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-oracle", type=float, default=0.3)
    ap.add_argument("--init", default=os.path.join(DATA, "policy.pt"))
    ap.add_argument("--out", default=os.path.join(DATA, "policy_ei.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    insts, oracle = load_train_pool()
    names = [n for n in insts if n in oracle]
    print(f"pool: {len(names)} instances with oracle data")

    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    model = EdgePolicy(ckpt["hidden"], ckpt["layers"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # fixed validation subset (held-out instances, unseen seeds)
    from tnco.instances import random_regular
    val_insts = [random_regular(60, 3, seed=5000 + s) for s in range(4)] + [
        random_regular(80, 3, seed=5000 + s) for s in range(4)
    ]

    for rnd in range(args.rounds):
        batch_names = random.sample(names, args.batch_instances)
        batch = [insts[n] for n in batch_names]
        results = rollout_batch(model, batch, args.k, device, args.temp, gen)
        # pick best trajectory per instance
        best_actions, best_tc = [], []
        for res in results:
            tc, acts = min(res, key=lambda x: x[0])
            best_tc.append(tc)
            best_actions.append(acts)
        model.train()
        opt.zero_grad()

        def chunked_loss(binstances, bactions, weight, chunk=8):
            total = 0.0
            for i in range(0, len(binstances), chunk):
                l = trajectory_ce_loss(
                    model, binstances[i : i + chunk], bactions[i : i + chunk], device
                )
                (l * weight / (len(binstances) / chunk)).backward()
                total += l.item()
            return total / max(1, len(binstances) // chunk)

        loss_pi = chunked_loss(batch, best_actions, 1.0)
        oracle_insts = [insts[n] for n in batch_names]
        oracle_acts = [oracle[n][0] for n in batch_names]
        loss_or = chunked_loss(oracle_insts, oracle_acts, args.w_oracle)
        loss = loss_pi + args.w_oracle * loss_or
        opt.step()
        print(
            f"round {rnd:3d} loss {loss:.4f} "
            f"mean best tc {sum(best_tc) / len(best_tc):.3f}",
            flush=True,
        )
        if (rnd + 1) % 10 == 0:
            model.eval()
            vres = rollout_batch(model, val_insts, 8, device, args.temp, gen)
            vtc = [min(r, key=lambda x: x[0])[0] for r in vres]
            print(f"  [val] best-of-8 mean tc: {sum(vtc) / len(vtc):.3f} "
                  f"(per-instance: {[round(v, 2) for v in vtc]})", flush=True)
            torch.save(
                {"state_dict": model.state_dict(), "hidden": ckpt["hidden"],
                 "layers": ckpt["layers"]},
                args.out,
            )
    torch.save(
        {"state_dict": model.state_dict(), "hidden": ckpt["hidden"],
         "layers": ckpt["layers"]},
        args.out,
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()

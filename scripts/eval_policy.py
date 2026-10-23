"""Evaluate the imitation policy: best-of-k sampled rollout tc vs TreeSA.

Usage: .venv/bin/python scripts/eval_policy.py [--k 50] [--temp 1.0]
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnco.contraction import ContractionState
from tnco.gnn import EdgePolicy, featurize
from tnco.instances import Instance, erdos_renyi, random_regular, square_lattice
from tnco.treesa import run_optimizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_PATH = os.path.join(ROOT, "data", "test_refs.json")


def test_instances():
    insts = []
    for s in range(5):
        insts.append(random_regular(60, 3, seed=5000 + s))   # in-distribution
        insts.append(random_regular(80, 3, seed=5000 + s))   # in-distribution
    for s in [10, 11]:
        insts.append(random_regular(100, 3, seed=s))          # size transfer
    insts.append(square_lattice(8, 8))                        # family transfer
    insts.append(erdos_renyi(100, 0.05, seed=7))              # family transfer
    insts.append(random_regular(100, 3, chi=16, seed=5))      # extent transfer
    return insts


def replay_tc(inst: Instance, actions: list[int]) -> float:
    """Exact log2 total FLOPs of an action sequence (integer arithmetic)."""
    state = ContractionState.initial(inst)
    flops = 0
    for a in actions:
        ta, tb = (state.tensors[i] for i in state.incidence[a])
        step = 1
        for ix in ta | tb:
            step *= state.size_dict[ix]
        flops += step
        state = state.contract(a)
    return math.log2(flops)


@torch.no_grad()
def rollout(model, inst: Instance, device: str, temp: float, greedy: bool):
    state = ContractionState.initial(inst)
    actions = []
    while not state.done():
        nf, ef, eff, labels = featurize(state)
        logits = model(
            torch.tensor(nf, dtype=torch.float32, device=device),
            torch.tensor(ef, dtype=torch.long, device=device),
            torch.tensor(eff, dtype=torch.float32, device=device),
            len(nf),
        )
        if greedy:
            pos = int(logits.argmax().item())
        else:
            pos = int(torch.multinomial(torch.softmax(logits / temp, dim=0), 1).item())
        actions.append(labels[pos])
        state = state.contract(labels[pos])
    return replay_tc(inst, actions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "data", "policy.pt"))
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--temp", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = EdgePolicy(ckpt["hidden"], ckpt["layers"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    refs = json.load(open(REF_PATH)) if os.path.exists(REF_PATH) else {}

    print(f"{'instance':30s} {'greedy':>8s} {'best-of-k':>9s} {'treesa':>8s} {'gap%':>6s}")
    for inst in test_instances():
        if inst.name not in refs:
            res = run_optimizer(inst, "treesa", seed=42, ntrials=10, niters=50)
            refs[inst.name] = res["tc"]
            json.dump(refs, open(REF_PATH, "w"), indent=1)
        tc_greedy = rollout(model, inst, device, args.temp, greedy=True)
        best = min(
            tc_greedy,
            min(rollout(model, inst, device, args.temp, False) for _ in range(args.k)),
        )
        ref = refs[inst.name]
        gap = (2 ** (best - ref) - 1) * 100
        print(f"{inst.name:30s} {tc_greedy:8.3f} {best:9.3f} {ref:8.3f} {gap:6.1f}", flush=True)


if __name__ == "__main__":
    main()

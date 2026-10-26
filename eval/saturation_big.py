"""Saturation probe at larger n / harder families (non-protected instances
in data/instances/big/). Appends to eval/results/saturation_big.jsonl."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnco.cost import contraction_complexity
from tnco.instances import Instance
from tnco.treesa import run_optimizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGETS = [(5, 50), (10, 100), (20, 200), (40, 400)]
INSTANCES = [
    "rr_n500_d3_c2_s0", "er_n200_p0.05_c2_s0", "rr_n500_d5_c2_s0",
    "rr_n500_d3_c16_s0", "rr_n1000_d3_c2_s0",
]


def main() -> None:
    out_path = os.path.join(ROOT, "eval", "results", "saturation_big.jsonl")
    for name in INSTANCES:
        inst = Instance.load(os.path.join(ROOT, "data", "instances", "big", name + ".json"), name=name)
        best = float("inf")
        for ntrials, niters in BUDGETS:
            res = run_optimizer(inst, "treesa", seed=7, ntrials=ntrials, niters=niters, timeout=14400)
            tc, sc, _ = contraction_complexity(inst, res["tree"])
            best = min(best, tc)
            with open(out_path, "a") as f:
                f.write(json.dumps({"instance": name, "ntrials": ntrials, "niters": niters,
                                    "work": ntrials * niters, "tc": tc, "best": best,
                                    "sc": sc, "elapsed": res["elapsed_seconds"]}) + "\n")
            print(f"{name:24s} W={ntrials * niters:6d} tc={tc:.4f} best={best:.4f} "
                  f"t={res['elapsed_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

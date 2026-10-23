"""TreeSA budget-saturation curves: best tc vs. budget (ntrials, niters).

Runs escalating budgets on a subset of main-track instances. Records to
eval/results/saturation.jsonl. Question: does TreeSA still improve with
more budget at n=100-200, or has it saturated (=> quality headroom for
any method is small)?
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnco.cost import contraction_complexity
from tnco.instances import Instance
from tnco.treesa import run_optimizer

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

# (ntrials, niters); total work = ntrials * niters
BUDGETS = [(10, 50), (20, 100), (40, 200), (80, 400)]

INSTANCES = [
    "main__rr_n100_d3_c2_s0",
    "main__rr_n200_d3_c2_s0",
    "main__rr_n100_d3_c16_s0",
    "main__er_n100_p0.05_c2_s0",
    "main__lat_8x8_c2",
]


def main() -> None:
    out_path = os.path.join(EVAL_DIR, "results", "saturation.jsonl")
    for name in INSTANCES:
        inst = Instance.load(os.path.join(EVAL_DIR, "instances", name + ".json"), name=name)
        best = float("inf")
        for ntrials, niters in BUDGETS:
            res = run_optimizer(inst, "treesa", seed=7, ntrials=ntrials, niters=niters, timeout=7200)
            tc, sc, rwc = contraction_complexity(inst, res["tree"])  # verified
            best = min(best, tc)
            work = ntrials * niters
            rec = {
                "instance": name, "ntrials": ntrials, "niters": niters,
                "work": work, "tc": tc, "best_tc_so_far": best,
                "sc": sc, "elapsed_seconds": res["elapsed_seconds"],
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{name:30s} ntrials={ntrials:3d} niters={niters:3d} "
                  f"work={work:6d} tc={tc:.4f} best={best:.4f} t={res['elapsed_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Run an optimizer over the protected suite; verify and record results.

Usage:
  .venv/bin/python eval/run_eval.py OPTIMIZER [--seed SEED] [--ntrials N] [--niters N]

Results are appended to eval/results/<optimizer>_seed<seed>.jsonl.
Every returned tree is verified with the independent exact checker;
mismatches are recorded as checker_ok=false (and should never happen).
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnco.cost import check_against_julia
from tnco.instances import Instance
from tnco.treesa import run_optimizer

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("optimizer", choices=["treesa", "greedy", "exact"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ntrials", type=int, default=10)
    ap.add_argument("--niters", type=int, default=50)
    ap.add_argument("--track", default="all", choices=["all", "exact", "main"])
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(EVAL_DIR, "instances", "*.json")))
    if args.track != "all":
        paths = [p for p in paths if os.path.basename(p).startswith(args.track)]

    out_dir = os.path.join(EVAL_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"{args.optimizer}_seed{args.seed}.jsonl"
    )

    with open(out_path, "a") as out_f:
        for path in paths:
            name = os.path.basename(path)[: -len(".json")]
            inst = Instance.load(path, name=name)
            if args.optimizer == "exact" and len(inst.inputs) > 22:
                continue
            res = run_optimizer(
                inst,
                args.optimizer,
                seed=args.seed,
                ntrials=args.ntrials,
                niters=args.niters,
            )
            ok, msg = check_against_julia(inst, res)
            rec = {
                "instance": name,
                "n_tensors": len(inst.inputs),
                "optimizer": args.optimizer,
                "seed": args.seed,
                "ntrials": args.ntrials,
                "niters": args.niters,
                "tc": res["tc"],
                "sc": res["sc"],
                "rwc": res["rwc"],
                "elapsed_seconds": res["elapsed_seconds"],
                "checker_ok": ok,
                "tree": res["tree"],
            }
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            print(
                f"{name:34s} n={len(inst.inputs):3d} tc={res['tc']:8.4f} "
                f"sc={res['sc']:5.1f} t={res['elapsed_seconds']:6.1f}s ok={ok}"
            )
            if not ok:
                print("   CHECKER MISMATCH:", msg)


if __name__ == "__main__":
    main()

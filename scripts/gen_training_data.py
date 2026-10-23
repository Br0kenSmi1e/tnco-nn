"""Generate the imitation training set.

Distribution: random regular graphs (d=3, chi=2) with n in {30..80},
plus a smaller number of lattices and ER graphs for diversity.
Oracle: Julia TreeSA (ntrials=5, niters=100), sharded across cores.

Outputs:
  data/instances/train/*.json     instance files
  data/oracle/train/*.json        oracle results (verified)
  data/train_actions.jsonl        {"instance": name, "tc": ..., "actions": [...]}

Usage: .venv/bin/python scripts/gen_training_data.py [--n-per-size 40]
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnco.instances import erdos_renyi, random_regular, square_lattice
from tnco.oracle import extract_actions, run_oracle_shard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_WORKERS = 14


def build_instances(n_per_size: int) -> list:
    insts = []
    for n in [30, 40, 50, 60, 70, 80]:
        for s in range(n_per_size):
            insts.append(random_regular(n, 3, seed=1000 + s))
    # diversity: some d=4, some chi variation, lattices, ER
    for s in range(n_per_size // 2):
        insts.append(random_regular(60, 4, seed=2000 + s))
        insts.append(random_regular(60, 3, chi=8, seed=3000 + s))
        insts.append(erdos_renyi(60, 0.06, seed=4000 + s))
    for side in [4, 5, 6]:
        for s in range(n_per_size // 3):
            insts.append(square_lattice(side, side, chi=2))
            insts.append(square_lattice(side, side + 1, chi=2))
    return insts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-size", type=int, default=40)
    args = ap.parse_args()

    inst_dir = os.path.join(ROOT, "data", "instances", "train")
    out_dir = os.path.join(ROOT, "data", "oracle", "train")
    os.makedirs(inst_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    insts = build_instances(args.n_per_size)
    # uniquify names (lattice generator names collide across s)
    seen = {}
    jobs = []
    for inst in insts:
        k = seen.get(inst.name, 0)
        seen[inst.name] = k + 1
        if k:
            inst.name = f"{inst.name}_r{k}"
        ipath = os.path.join(inst_dir, inst.name + ".json")
        opath = os.path.join(out_dir, inst.name + ".json")
        if not os.path.exists(ipath):
            inst.save(ipath)
        if not os.path.exists(opath):
            jobs.append(
                {
                    "instance": ipath,
                    "out": opath,
                    "ntrials": 5,
                    "niters": 100,
                    "seed": 17,
                }
            )

    print(f"{len(insts)} instances, {len(jobs)} oracle jobs to run")
    shards = [jobs[i::N_WORKERS] for i in range(N_WORKERS)]
    with cf.ThreadPoolExecutor(N_WORKERS) as ex:
        list(ex.map(run_oracle_shard, [s for s in shards if s]))

    # extract action sequences
    out_path = os.path.join(ROOT, "data", "train_actions.jsonl")
    n_ok = n_disjoint = n_missing = 0
    with open(out_path, "w") as f:
        for inst in insts:
            opath = os.path.join(out_dir, inst.name + ".json")
            if not os.path.exists(opath):
                n_missing += 1
                continue
            with open(opath) as g:
                res = json.load(g)
            actions = extract_actions(inst, res)
            if actions is None:
                n_disjoint += 1
                continue
            f.write(
                json.dumps(
                    {
                        "instance": inst.name,
                        "n_tensors": len(inst.inputs),
                        "oracle_tc": res["tc"],
                        "actions": actions,
                    }
                )
                + "\n"
            )
            n_ok += 1
    print(f"wrote {n_ok} action sequences to {out_path} "
          f"(disjoint-merge skips: {n_disjoint}, missing: {n_missing})")


if __name__ == "__main__":
    main()

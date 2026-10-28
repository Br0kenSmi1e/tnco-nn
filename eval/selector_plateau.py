"""Selector plateau study: does selector advantage persist with budget?

Selectors {random, greedy1, eps_greedy(0.3)} x W {250, 1000, 4000}
x seeds {21,22,23} on rr_n500_d3 and er_n200 (the two instances where
the selector mattered at W=250). Question: same plateau (fast convergence)
or persistent advantage (better plateau)?
"""

import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tnco.cost import check_against_julia
from tnco.instances import Instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "eval", "results", "selector_plateau.jsonl")
DONE = set()
if os.path.exists(OUT):
    for line in open(OUT):
        r = json.loads(line)
        DONE.add((r["instance"], r["selector"], r["work"], r["seed"]))


def run(inst_path, selector, ntrials, niters, seed, timeout=21600):
    inst = Instance.load(inst_path)
    with tempfile.TemporaryDirectory() as td:
        op = os.path.join(td, "o.json")
        cmd = ["julia", f"--project={ROOT}/julia", f"{ROOT}/julia/sa_selector.jl",
               inst_path, op, selector, str(ntrials), str(niters), str(seed)]
        if selector == "eps_greedy":
            cmd.append("0.3")
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-1500:])
        res = json.load(open(op))
        ok, msg = check_against_julia(inst, res)
        assert ok, msg
        return res["tc"]


def main():
    jobs = []
    for name in ["rr_n500_d3_c2_s0", "er_n200_p0.05_c2_s0"]:
        ip = os.path.join(ROOT, "data", "instances", "big", name + ".json")
        for selector in ["random", "greedy1", "eps_greedy"]:
            for ntrials, niters in [(5, 50), (10, 100), (20, 200)]:
                for seed in [21, 22, 23]:
                    if (name, selector, ntrials * niters, seed) not in DONE:
                        jobs.append((name, ip, selector, ntrials, niters, seed))
    print(f"{len(jobs)} runs to do", flush=True)
    lock = __import__("threading").Lock()

    def do(job):
        name, ip, selector, ntrials, niters, seed = job
        tc = run(ip, selector, ntrials, niters, seed)
        rec = {"instance": name, "selector": selector, "work": ntrials * niters,
               "ntrials": ntrials, "niters": niters, "seed": seed, "tc": tc}
        with lock:
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"{name:22s} {selector:11s} W={ntrials*niters:5d} seed={seed} tc={tc:.3f}",
              flush=True)

    with cf.ThreadPoolExecutor(12) as ex:
        list(ex.map(do, jobs))


if __name__ == "__main__":
    main()

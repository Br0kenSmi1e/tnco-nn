"""Selector probe: does the SA rule selector matter on unsaturated landscapes?"""
import concurrent.futures as cf
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tnco.instances import Instance
from tnco.cost import check_against_julia

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_selector(inst_path, selector, ntrials, niters, seed, eps=None, timeout=10800):
    inst = Instance.load(inst_path)
    with tempfile.TemporaryDirectory() as td:
        op = os.path.join(td, "o.json")
        cmd = ["julia", f"--project={ROOT}/julia", f"{ROOT}/julia/sa_selector.jl",
               inst_path, op, selector, str(ntrials), str(niters), str(seed)]
        if eps is not None:
            cmd.append(str(eps))
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-1500:])
        res = json.load(open(op))
        ok, msg = check_against_julia(inst, res)
        assert ok, msg
        return res["tc"]

def main():
    jobs = []
    for name in ["rr_n500_d3_c2_s0", "rr_n500_d5_c2_s0", "er_n200_p0.05_c2_s0"]:
        ip = os.path.join(ROOT, "data", "instances", "big", name + ".json")
        for selector, eps in [("random", None), ("eps_greedy", 0.3)]:
            for seed in [21, 22, 23]:
                jobs.append((name, ip, selector, 5, 50, seed, eps))
    with cf.ThreadPoolExecutor(14) as ex:
        futs = {ex.submit(run_selector, ip, sel, nt, ni, s, e): (name, sel, s)
                for name, ip, sel, nt, ni, s, e in jobs}
        results = {}
        for f in cf.as_completed(futs):
            name, sel, s = futs[f]
            tc = f.result()
            results.setdefault((name, sel), []).append(tc)
            print(f"{name:22s} {sel:11s} seed={s} W=250 tc={tc:.3f}", flush=True)
    with open(os.path.join(ROOT, "eval", "results", "selector_probe.json"), "w") as f:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in results.items()}, f, indent=1)

if __name__ == "__main__":
    main()

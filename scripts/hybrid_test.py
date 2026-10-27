"""Hybrid experiment: policy-init SA vs greedy/random-init SA, matched budget."""
import json, sys, torch
sys.path.insert(0, ".")
from tnco.instances import Instance
from tnco.contraction import actions_to_tree
from tnco.cost import check_against_julia
from tnco.gnn import EdgePolicy
from tnco.rollout import rollout_batch
from tnco.treesa import run_optimizer

device = "cuda"
ckpt = torch.load("data/policy_ei.pt", map_location=device, weights_only=False)
model = EdgePolicy(ckpt["hidden"], ckpt["layers"]).to(device)
model.load_state_dict(ckpt["state_dict"]); model.eval()

for name in ["rr_n500_d3_c2_s0"]:
    inst = Instance.load(f"data/instances/big/{name}.json")
    res = rollout_batch(model, [inst], 50, device, 1.0)
    tc, actions = min(res[0], key=lambda x: x[0])
    tree = actions_to_tree(inst, actions)
    print(f"{name}: policy init tc={tc:.3f}", flush=True)
    for ntrials, niters in [(5, 50), (10, 100)]:
        tcs = []
        for seed in [11, 12]:
            out = run_optimizer(inst, "treesa", seed=seed, ntrials=ntrials,
                                niters=niters, init_tree=tree, timeout=7200)
            ok, msg = check_against_julia(inst, out)
            assert ok, msg
            tcs.append(out["tc"])
        print(f"  policy-init SA W={ntrials*niters}: {['%.3f' % t for t in tcs]}", flush=True)

"""Wrapper for the Julia optimizer runner (julia/run_optimizer.jl)."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

from .instances import Instance

JULIA_DIR = os.path.join(os.path.dirname(__file__), "..", "julia")


def run_optimizer(
    inst: Instance,
    optimizer: str = "treesa",
    seed: int = 42,
    ntrials: int = 10,
    niters: int = 50,
    timeout: float = 3600.0,
    initializer: str = "greedy",
    init_tree: list | int | None = None,
) -> dict:
    """Run a Julia optimizer; returns the result dict with keys
    tc, sc, rwc, tree, optimizer, seed, ntrials, niters, elapsed_seconds.

    optimizer ∈ {"treesa", "greedy", "exact"}; treesa uses a pure-tc scorer.
    """
    with tempfile.TemporaryDirectory() as td:
        inst_path = os.path.join(td, "instance.json")
        out_path = os.path.join(td, "result.json")
        inst.save(inst_path)
        args_list = [
            "julia",
            f"--project={os.path.abspath(JULIA_DIR)}",
            os.path.join(os.path.abspath(JULIA_DIR), "run_optimizer.jl"),
            inst_path,
            out_path,
            optimizer,
            str(seed),
            str(ntrials),
            str(niters),
        ]
        if init_tree is not None:
            init_path = os.path.join(td, "init_tree.json")
            with open(init_path, "w") as f:
                json.dump(init_tree, f)
            args_list += ["specified", init_path]
        else:
            args_list.append(initializer)
        proc = subprocess.run(args_list, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"{optimizer} failed:\n{proc.stderr}")
        with open(out_path) as f:
            return json.load(f)


def run_treesa(
    inst: Instance,
    ntrials: int = 10,
    niters: int = 50,
    seed: int = 42,
    timeout: float = 3600.0,
) -> dict:
    """Backward-compatible TreeSA wrapper."""
    return run_optimizer(
        inst, "treesa", seed=seed, ntrials=ntrials, niters=niters, timeout=timeout
    )

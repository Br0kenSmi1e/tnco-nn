"""Oracle training-data generation: instances -> TreeSA trees -> action sequences.

Pipeline:
  1. generate training instances (tnco.instances generators)
  2. run Julia TreeSA on them, sharded across cores (gen_oracle.jl)
  3. extract action sequences via linearize_tree; trees with disjoint
     merges are retried once with a different seed, then skipped
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

from .contraction import linearize_tree
from .cost import check_against_julia
from .instances import Instance

JULIA_DIR = os.path.join(os.path.dirname(__file__), "..", "julia")


def run_oracle_shard(jobs: list[dict]) -> None:
    """Run one shard of oracle jobs in a single Julia process."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False
    ) as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")
        jobfile = f.name
    try:
        proc = subprocess.run(
            [
                "julia",
                f"--project={os.path.abspath(JULIA_DIR)}",
                os.path.join(os.path.abspath(JULIA_DIR), "gen_oracle.jl"),
                jobfile,
            ],
            capture_output=True,
            text=True,
            timeout=86400,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"gen_oracle failed:\n{proc.stderr[-3000:]}")
    finally:
        os.unlink(jobfile)


def extract_actions(
    inst: Instance, result: dict, tol: float = 1e-9
) -> list[int] | None:
    """Verify oracle result and linearize; None if the tree has disjoint
    merges (not representable in the edge-picking MDP)."""
    ok, msg = check_against_julia(inst, result, tol=tol)
    if not ok:
        raise ValueError(f"oracle checker mismatch: {msg}")
    try:
        return linearize_tree(inst, result["tree"])
    except ValueError:
        return None

"""Tensor-network instance representation and generators.

An instance is a closed tensor network (contracts to a scalar) given by:
  - inputs: list of index-label lists, one per tensor
  - size_dict: index label -> extent

Shared JSON format with julia/run_treesa.jl:
  {"inputs": [[1,2],[2,3],...], "output": [], "size_dict": {"1": 2, ...}}
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field


@dataclass
class Instance:
    inputs: list[list[int]]
    size_dict: dict[int, int]
    output: list[int] = field(default_factory=list)
    name: str = ""

    def to_json_dict(self) -> dict:
        return {
            "inputs": self.inputs,
            "output": self.output,
            "size_dict": {str(k): v for k, v in self.size_dict.items()},
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json_dict(), f)

    @classmethod
    def load(cls, path: str, name: str = "") -> "Instance":
        with open(path) as f:
            d = json.load(f)
        return cls(
            inputs=[list(ix) for ix in d["inputs"]],
            output=list(d["output"]),
            size_dict={int(k): int(v) for k, v in d["size_dict"].items()},
            name=name or path,
        )


def _from_edges(
    n: int, edges: list[tuple[int, int]], chi: int, name: str
) -> Instance:
    """Build an instance from a simple graph: tensors = vertices, one index
    per graph edge, shared by its two endpoint tensors. Extent chi everywhere.
    """
    inputs: list[list[int]] = [[] for _ in range(n)]
    size_dict: dict[int, int] = {}
    for e, (u, v) in enumerate(edges):
        inputs[u].append(e)
        inputs[v].append(e)
        size_dict[e] = chi
    return Instance(inputs=inputs, size_dict=size_dict, name=name)


def random_regular(n: int, degree: int, chi: int = 2, seed: int = 0) -> Instance:
    """Random regular graph on n tensors (n*degree must be even)."""
    rng = random.Random(seed)
    while True:
        stubs = [v for v in range(n) for _ in range(degree)]
        rng.shuffle(stubs)
        edges = []
        ok = True
        seen = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or (min(u, v), max(u, v)) in seen:
                ok = False
                break
            seen.add((min(u, v), max(u, v)))
            edges.append((u, v))
        if ok:
            break
    return _from_edges(n, edges, chi, f"rr_n{n}_d{degree}_c{chi}_s{seed}")


def square_lattice(lx: int, ly: int, chi: int = 2) -> Instance:
    """Square lattice grid (periodic in neither direction)."""

    def vid(x: int, y: int) -> int:
        return y * lx + x

    edges = []
    for y in range(ly):
        for x in range(lx):
            if x + 1 < lx:
                edges.append((vid(x, y), vid(x + 1, y)))
            if y + 1 < ly:
                edges.append((vid(x, y), vid(x, y + 1)))
    return _from_edges(lx * ly, edges, chi, f"lat_{lx}x{ly}_c{chi}")


def erdos_renyi(n: int, p: float, chi: int = 2, seed: int = 0) -> Instance:
    """Erdos-Renyi G(n, p); isolated vertices get a self-loop-free reconnect
    by rejection (resample until no isolated vertices)."""
    rng = random.Random(seed)
    while True:
        edges = [
            (u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p
        ]
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if all(d > 0 for d in deg):
            break
    return _from_edges(n, edges, chi, f"er_n{n}_p{p}_c{chi}_s{seed}")

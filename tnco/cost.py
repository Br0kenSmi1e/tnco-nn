"""Independent exact complexity checker.

Recomputes tc/sc/rwc from an instance and a contraction tree using exact
integer arithmetic, independently of any optimizer. Conventions (matching
OMEinsumContractionOrders' contraction_complexity):

  - A contraction step merges two tensors a, b into out = a ∪ b (index sets).
    step FLOPs      = prod(extents of indices in out)
    step read-write = elements(a) + elements(b) + elements(out)
  - tc  = log2(sum of step FLOPs over all steps)
  - sc  = log2(max elements of any contraction-step OUTPUT, i.e. the
          largest intermediate/result tensor; input tensor sizes are NOT
          counted, matching OMEinsumContractionOrders' convention)
  - rwc = log2(sum of step read-writes over all steps)

Tree format (shared): leaf = tensor index (int); node = [left, right].
"""

from __future__ import annotations

import math

from .instances import Instance


def _elements(indices: list[int], size_dict: dict[int, int]) -> int:
    n = 1
    for ix in indices:
        n *= size_dict[ix]
    return n


def contraction_complexity(
    inst: Instance, tree: list | int
) -> tuple[float, float, float]:
    """Exact (tc, sc, rwc) in log2. Raises on malformed trees."""
    n = len(inst.inputs)
    # incidence[ix] = set of tensor ids containing index ix
    incidence: dict[int, set[int]] = {}
    for t, ixs in enumerate(inst.inputs):
        for ix in ixs:
            incidence.setdefault(ix, set()).add(t)
    output_set = set(inst.output)

    def walk(node) -> tuple[frozenset[int], frozenset[int], int, int, int]:
        # returns (live indices, leaf set, flops_sum, rw_sum, max_elems)
        if isinstance(node, int):
            if not (0 <= node < n):
                raise ValueError(f"leaf index {node} out of range")
            idx = frozenset(inst.inputs[node])
            elems = _elements(inst.inputs[node], inst.size_dict)
            return idx, frozenset([node]), 0, 0, elems
        if not (isinstance(node, list) and len(node) == 2):
            raise ValueError(f"tree node must be [left, right], got {node!r}")
        la, ta, fa, ra, ma = walk(node[0])
        lb, tb, fb, rb, mb = walk(node[1])
        if ta & tb:
            raise ValueError("tree uses a tensor more than once")
        leaves = ta | tb
        u = la | lb
        # an index survives iff some incident tensor lies outside this
        # subtree, or it is an open (output) index
        live = frozenset(
            ix
            for ix in u
            if ix in output_set or not incidence[ix] <= leaves
        )
        step_flops = _elements(sorted(u), inst.size_dict)
        step_rw = (
            _elements(sorted(la), inst.size_dict)
            + _elements(sorted(lb), inst.size_dict)
            + _elements(sorted(live), inst.size_dict)
        )
        out_elems = _elements(sorted(live), inst.size_dict)
        # leaf sizes (ma, mb from leaves) are excluded from sc: only
        # contraction-step outputs count
        step_out = out_elems
        child_max = max(
            (x for x, is_leaf in [(ma, len(ta) == 1), (mb, len(tb) == 1)] if not is_leaf),
            default=0,
        )
        return (
            live,
            leaves,
            fa + fb + step_flops,
            ra + rb + step_rw,
            max(child_max, step_out),
        )

    live, leaves, flops, rw, max_elems = walk(tree)
    if len(leaves) != n:
        raise ValueError("tree does not use every tensor exactly once")
    if set(live) != output_set:
        raise ValueError("tree output indices do not match instance output")
    # sc: max over contraction-step outputs only (inputs excluded)
    if n == 1:
        max_elems = _elements(inst.output, inst.size_dict)
    log2 = lambda x: math.log2(x) if x > 0 else float("-inf")
    return log2(flops), log2(max_elems), log2(rw)


def check_against_julia(
    inst: Instance, result: dict, tol: float = 1e-9
) -> tuple[bool, str]:
    """Negative control: our checker must reproduce Julia's reported
    complexities for Julia's own tree."""
    tc, sc, rwc = contraction_complexity(inst, result["tree"])
    ok = (
        abs(tc - result["tc"]) <= tol
        and abs(sc - result["sc"]) <= tol
        and abs(rwc - result["rwc"]) <= tol
    )
    msg = (
        f"checker=({tc:.9f}, {sc:.9f}, {rwc:.9f}) "
        f"julia=({result['tc']:.9f}, {result['sc']:.9f}, {result['rwc']:.9f})"
    )
    return ok, msg

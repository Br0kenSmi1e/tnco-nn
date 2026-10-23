"""Contraction-state simulation and oracle linearization.

The sequential-contraction MDP: state = current set of tensors (each a
frozenset of live index labels); action = pick an index (edge) whose two
incident tensors get merged. Merging eliminates exactly the indices whose
incident tensors are all being merged (and which are not open/output).

Also: linearize an oracle binary tree into an equivalent action sequence
(bottom-up), which is what imitation learning consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .instances import Instance


@dataclass
class ContractionState:
    """Current contraction state of an instance."""

    size_dict: dict[int, int]
    tensors: list[frozenset[int]]  # live tensors (index sets)
    # index -> tuple of current tensor ids containing it
    incidence: dict[int, tuple[int, ...]]
    output: frozenset[int]

    @classmethod
    def initial(cls, inst: Instance) -> "ContractionState":
        incidence: dict[int, list[int]] = {}
        for t, ixs in enumerate(inst.inputs):
            for ix in set(ixs):
                incidence.setdefault(ix, []).append(t)
        return cls(
            size_dict=inst.size_dict,
            tensors=[frozenset(ixs) for ixs in inst.inputs],
            incidence={k: tuple(v) for k, v in incidence.items()},
            output=frozenset(inst.output),
        )

    def contract(self, index: int) -> "ContractionState":
        """Merge the tensors sharing `index`; eliminate dead indices."""
        t_ids = self.incidence[index]
        if len(t_ids) != 2:
            raise ValueError(
                f"index {index} is incident to {len(t_ids)} tensors "
                "(only simple-graph indices are contractible actions)"
            )
        a, b = t_ids
        ta, tb = self.tensors[a], self.tensors[b]
        merged = ta | tb
        # surviving tensors: all except a, b, plus the merged one
        keep = [t for i, t in enumerate(self.tensors) if i not in (a, b)]
        # an index in merged is eliminated iff all its incident tensors
        # are in {a, b} and it is not an open index
        live = frozenset(
            ix
            for ix in merged
            if ix in self.output
            or any(t not in (a, b) for t in self.incidence[ix])
        )
        keep.append(live)
        # rebuild incidence
        incidence: dict[int, list[int]] = {}
        for t, ixs in enumerate(keep):
            for ix in ixs:
                incidence.setdefault(ix, []).append(t)
        return ContractionState(
            size_dict=self.size_dict,
            tensors=keep,
            incidence={k: tuple(v) for k, v in incidence.items()},
            output=self.output,
        )

    def contractible_indices(self) -> list[int]:
        """Indices incident to exactly two distinct tensors."""
        return [
            ix
            for ix, ts in self.incidence.items()
            if len(ts) == 2 and ts[0] != ts[1]
        ]

    def done(self) -> bool:
        return len(self.tensors) == 1


def linearize_tree(inst: Instance, tree: list | int) -> list[int]:
    """Convert an oracle binary tree (shared nested-list format) into an
    equivalent index-elimination sequence for the sequential MDP.

    Canonical order: repeatedly contract the pair of current tensors
    corresponding to the deepest unprocessed internal node (post-order).
    Raises ValueError on disjoint (index-free) merges.
    """
    n = len(inst.inputs)

    # post-order list of internal nodes; each node: (left_key, right_key)
    # subtree keys: ("L", tensor_id) for leaves, ("N", node_id) for internal
    nodes: list[tuple] = []

    def post(node) -> tuple:
        if isinstance(node, int):
            return ("L", node)
        lid = post(node[0])
        rid = post(node[1])
        nodes.append((lid, rid))
        return ("N", len(nodes) - 1)

    root = post(tree)
    if root[0] == "L":  # single-tensor network
        return []

    # replay keyed by subtree id (positions shift after each merge, so we
    # track tensors by key, not by list position)
    output_set = frozenset(inst.output)
    cur: dict[tuple, frozenset[int]] = {
        ("L", i): frozenset(inst.inputs[i]) for i in range(n)
    }
    actions: list[int] = []

    # process internal nodes in post-order (children before parents)
    for node_id, (lk, rk) in enumerate(nodes):
        a = cur.pop(lk)
        b = cur.pop(rk)
        shared = a & b
        if not shared:
            raise ValueError("oracle tree contains a disjoint (outer-product) merge")
        # canonical choice among shared indices: largest extent, then label
        index = max(shared, key=lambda ix: (inst.size_dict[ix], ix))
        actions.append(index)
        merged = a | b
        live = frozenset(
            ix
            for ix in merged
            if ix in output_set or any(ix in s for s in cur.values())
        )
        cur[("N", node_id)] = live
    return actions

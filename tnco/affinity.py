"""Affinity learnability probe.

Target: from an oracle action sequence, the *merge time* m_ij = step at
which original tensors i and j first land in the same cluster (normalized
by n-1). Question: is m_ij predictable from the instance graph?

Model: GNN over the static instance graph -> node embeddings -> pair
decoder MLP -> predicted merge time. Controls: graph shortest-path
distance and degree features (if the GNN doesn't beat those, the learned
embedding adds nothing).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .instances import Instance


def merge_times(inst: Instance, actions: list[int]) -> torch.Tensor:
    """[n, n] tensor of normalized merge steps; -1 diagonal.

    Replays the oracle action sequence with ContractionState (verified
    index bookkeeping) while tracking each current tensor's cluster of
    original tensor ids."""
    from .contraction import ContractionState

    n = len(inst.inputs)
    state = ContractionState.initial(inst)
    clusters: list[frozenset[int]] = [frozenset([i]) for i in range(n)]
    m = torch.full((n, n), -1.0)
    for step, a in enumerate(actions):
        x, y = state.incidence[a]
        cx, cy = clusters[x], clusters[y]
        for i in cx:
            for j in cy:
                m[i, j] = m[j, i] = step
        clusters = [c for p, c in enumerate(clusters) if p not in (x, y)]
        clusters.append(cx | cy)
        state = state.contract(a)
    return m / max(n - 1, 1)


def lca_sizes(inst: Instance, tree) -> torch.Tensor:
    """[n, n] tensor: for each pair (i, j), the number of leaves of the
    smallest subtree containing both (the LCA cluster size), normalized
    by n. Linearization-invariant property of the oracle tree."""
    n = len(inst.inputs)
    m = torch.full((n, n), -1.0)

    def leaves_of(node) -> frozenset:
        if isinstance(node, int):
            return frozenset([node])
        return leaves_of(node[0]) | leaves_of(node[1])

    # collect (leaf_set) of every subtree, smallest-first
    sets = []

    def walk(node):
        if isinstance(node, int):
            return frozenset([node])
        s = walk(node[0]) | walk(node[1])
        sets.append(s)
        return s

    walk(tree)
    sets.sort(key=len)
    for s in sets:
        ls = list(s)
        for a in range(len(ls)):
            for b in range(a + 1, len(ls)):
                if m[ls[a], ls[b]] < 0:
                    m[ls[a], ls[b]] = m[ls[b], ls[a]] = len(s)
    return m / n


def static_graph(inst: Instance):
    """Node features + edge list of the instance (tensors = nodes).
    Node feat: [log2 size, degree, mean log2 extent]; edge feat: [log2 extent]."""
    log2s = {ix: math.log2(s) for ix, s in inst.size_dict.items()}
    nf = []
    for ixs in inst.inputs:
        ls = [log2s[i] for i in set(ixs)]
        nf.append([sum(ls), float(len(ls)), sum(ls) / max(len(ls), 1)])
    edges, ef = [], []
    incidence: dict[int, list[int]] = {}
    for t, ixs in enumerate(inst.inputs):
        for ix in set(ixs):
            incidence.setdefault(ix, []).append(t)
    for ix, ts in incidence.items():
        if len(ts) == 2:
            edges.append(ts)
            ef.append([log2s[ix]])
    return nf, edges, ef


def shortest_path_matrix(inst: Instance) -> torch.Tensor:
    """BFS distances on the instance graph (control)."""
    n = len(inst.inputs)
    adj = [set() for _ in range(n)]
    incidence: dict[int, list[int]] = {}
    for t, ixs in enumerate(inst.inputs):
        for ix in set(ixs):
            incidence.setdefault(ix, []).append(t)
    for ts in incidence.values():
        for a in ts:
            for b in ts:
                if a != b:
                    adj[a].add(b)
    import collections

    d = torch.full((n, n), -1.0)
    for s in range(n):
        dist = {s: 0}
        q = collections.deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for v, dv in dist.items():
            d[s, v] = dv
    return d


class AffinityNet(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 6, emb: int = 32, in_dim: int = 3):
        super().__init__()
        self.node_in = nn.Linear(in_dim, hidden)
        self.msg = nn.ModuleList(
            [nn.Linear(2 * hidden + 1, hidden) for _ in range(layers)]
        )
        self.upd = nn.ModuleList(
            [nn.Linear(2 * hidden, hidden) for _ in range(layers)]
        )
        self.emb = nn.Linear(hidden, emb)
        self.dec = nn.Sequential(
            nn.Linear(3 * emb, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, nf, edges, ef, pairs):
        """nf [N,3], edges [E,2], ef [E,1], pairs [P,2] -> [P] predictions."""
        h = F.relu(self.node_in(nf))
        src, dst = edges[:, 0], edges[:, 1]
        for k in range(len(self.msg)):
            m = F.relu(self.msg[k](torch.cat([h[src], h[dst], ef], dim=1)))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            agg.index_add_(0, src, m)
            h = h + F.relu(self.upd[k](torch.cat([h, agg], dim=1)))
        z = self.emb(h)
        pi, pj = pairs[:, 0], pairs[:, 1]
        feat = torch.cat([z[pi] * z[pj], (z[pi] - z[pj]).abs(), z[pi] + z[pj]], dim=1)
        return self.dec(feat).squeeze(-1)

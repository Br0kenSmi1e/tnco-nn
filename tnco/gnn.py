"""GNN policy over contraction states (pure PyTorch, no PyG).

State featurization of a ContractionState:
  node features (per live tensor):
    [log2_elems, degree, mean_log2_extent]
  edge features (per contractible index, endpoints u,v):
    [log2_extent, log2_step_cost, log2_elems_u, log2_elems_v, removed]
    where step_cost = elements of the merged output before elimination
    (i.e. log2 FLOPs of this step), removed = step_cost - elems_u - elems_v
    (log-space analogue of the opt_einsum greedy score).

Architecture: K rounds of message passing with edge-conditioned messages,
residual node updates, per-edge scalar readout -> softmax over edges.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contraction import ContractionState
from .instances import Instance

NODE_DIM = 3
EDGE_DIM = 5


def featurize(state: ContractionState):
    """Return (node_feat [N,3], edges [E,2] endpoint ids, edge_feat [E,5],
    edge_index_labels [E]) for all contractible indices."""
    log2_sizes = {ix: math.log2(s) for ix, s in state.size_dict.items()}
    node_feat = []
    for t in state.tensors:
        ls = [log2_sizes[ix] for ix in t]
        node_feat.append([sum(ls), float(len(ls)), sum(ls) / max(len(ls), 1)])
    edges, edge_feat, labels = [], [], []
    for ix, ts in state.incidence.items():
        if len(ts) != 2 or ts[0] == ts[1]:
            continue
        a, b = ts
        la = sum(log2_sizes[i] for i in state.tensors[a])
        lb = sum(log2_sizes[i] for i in state.tensors[b])
        step = sum(log2_sizes[i] for i in state.tensors[a] | state.tensors[b])
        edges.append((a, b))
        edge_feat.append([log2_sizes[ix], step, la, lb, step - la - lb])
        labels.append(ix)
    return node_feat, edges, edge_feat, labels


class EdgePolicy(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 6):
        super().__init__()
        self.layers = layers
        self.node_in = nn.Linear(NODE_DIM, hidden)
        self.edge_msg = nn.ModuleList(
            [nn.Linear(2 * hidden + EDGE_DIM, hidden) for _ in range(layers)]
        )
        self.node_up = nn.ModuleList(
            [nn.Linear(2 * hidden, hidden) for _ in range(layers)]
        )
        self.score = nn.Sequential(
            nn.Linear(3 * hidden + EDGE_DIM, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, node_feat, edges, edge_feat, num_nodes):
        """node_feat: [N,3]; edges: [E,2] int64; edge_feat: [E,5].
        Returns logits [E]."""
        h = self.node_in(node_feat)
        src, dst = edges[:, 0], edges[:, 1]
        for k in range(self.layers):
            m = F.relu(
                self.edge_msg[k](torch.cat([h[src], h[dst], edge_feat], dim=1))
            )
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            agg.index_add_(0, src, m)  # undirected
            h = h + F.relu(self.node_up[k](torch.cat([h, agg], dim=1)))
        sym = torch.cat(
            [h[src] + h[dst], (h[src] - h[dst]).abs(), h[src] * h[dst], edge_feat],
            dim=1,
        )
        return self.score(sym).squeeze(-1)

"""Batched policy rollouts on GPU + teacher-forced loss on trajectories.

Rollouts: all states' graphs are concatenated into one big graph for a
single EdgePolicy forward; per-state softmax/sampling uses a padded
[G, Emax] layout.

Loss: teacher-forced cross-entropy over (state, chosen action) steps of
given trajectories, using segment-wise log-softmax over one concatenated
forward pass.
"""

from __future__ import annotations

import math

import torch

from .contraction import ContractionState
from .gnn import featurize
from .instances import Instance


def _batch_forward(model, batch_states: list[ContractionState], device):
    """One forward for many states.

    Returns (padded_logits [G, Emax], mask [G, Emax], labels_per_state).
    """
    node_feats, edge_rows, edge_feats, labels_all = [], [], [], []
    g_ids_l, j_ids_l = [], []
    node_offset = 0
    for g, st in enumerate(batch_states):
        nf, ef, eff, labels = featurize(st)
        node_feats.extend(nf)
        for j, ((a, b), f) in enumerate(zip(ef, eff)):
            edge_rows.append((a + node_offset, b + node_offset))
            edge_feats.append(f)
            g_ids_l.append(g)
            j_ids_l.append(j)
        labels_all.append(labels)
        node_offset += len(nf)
    edges = torch.tensor(edge_rows, dtype=torch.long, device=device)
    nf_t = torch.tensor(node_feats, dtype=torch.float32, device=device)
    eff_t = torch.tensor(edge_feats, dtype=torch.float32, device=device)
    logits = model(nf_t, edges, eff_t, nf_t.shape[0])
    G = len(batch_states)
    Emax = max(len(l) for l in labels_all)
    g_ids = torch.tensor(g_ids_l, dtype=torch.long, device=device)
    j_ids = torch.tensor(j_ids_l, dtype=torch.long, device=device)
    padded = torch.full((G, Emax), float("-inf"), device=device)
    padded[g_ids, j_ids] = logits
    mask = torch.zeros(G, Emax, dtype=torch.bool, device=device)
    mask[g_ids, j_ids] = True
    return padded, mask, labels_all


@torch.no_grad()
def rollout_batch(
    model,
    instances: list[Instance],
    k: int,
    device: str,
    temp: float = 1.0,
    gen: torch.Generator | None = None,
) -> list[list[tuple[float, list[int]]]]:
    """For each instance, run k rollouts (rollout 0 of each instance is
    greedy). Returns per instance: list of (tc, actions)."""
    states, owners = [], []
    for i, inst in enumerate(instances):
        for _ in range(k):
            states.append(ContractionState.initial(inst))
            owners.append(i)
    actions_all = [[] for _ in states]
    flops = [0] * len(states)
    active = list(range(len(states)))
    while active:
        padded, mask, labels_all = _batch_forward(
            model, [states[r] for r in active], device
        )
        probs = torch.softmax(padded / temp, dim=1)
        probs = probs * mask  # zero out padding (softmax already -inf, be safe)
        sampled = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        greedy_choice = padded.argmax(dim=1)
        is_greedy = torch.tensor(
            [r % k == 0 for r in active], dtype=torch.bool, device=device
        )
        chosen = torch.where(is_greedy, greedy_choice, sampled)
        next_active = []
        for row, r in enumerate(active):
            st = states[r]
            pos = int(chosen[row].item())
            ix = labels_all[row][pos]
            ta, tb = (st.tensors[i] for i in st.incidence[ix])
            step = 1
            for e in ta | tb:
                step *= st.size_dict[e]
            flops[r] += step
            actions_all[r].append(ix)
            states[r] = st.contract(ix)
            if not states[r].done():
                next_active.append(r)
        active = next_active
    out: list[list[tuple[float, list[int]]]] = [[] for _ in instances]
    for r, i in enumerate(owners):
        out[i].append((math.log2(flops[r]), actions_all[r]))
    return out


def trajectory_ce_loss(
    model,
    instances: list[Instance],
    actions_list: list[list[int]],
    device: str,
) -> torch.Tensor:
    """Teacher-forced mean per-step cross-entropy over given trajectories.

    One concatenated forward; segment-wise log-softmax per step.
    """
    node_feats, edge_rows, edge_feats = [], [], []
    seg_ids_l, targets = [], []
    node_offset = 0
    seg = 0
    for inst, actions in zip(instances, actions_list):
        st = ContractionState.initial(inst)
        for a in actions:
            nf, ef, eff, labels = featurize(st)
            if a not in labels:
                break
            pos = labels.index(a)
            base = len(edge_rows)
            for (u, v), f in zip(ef, eff):
                edge_rows.append((u + node_offset, v + node_offset))
                edge_feats.append(f)
            seg_ids_l.extend([seg] * len(ef))
            targets.append(base + pos)
            node_feats.extend(nf)
            node_offset += len(nf)
            seg += 1
            st = st.contract(a)
    nf_t = torch.tensor(node_feats, dtype=torch.float32, device=device)
    edges = torch.tensor(edge_rows, dtype=torch.long, device=device)
    eff_t = torch.tensor(edge_feats, dtype=torch.float32, device=device)
    logits = model(nf_t, edges, eff_t, nf_t.shape[0])
    seg_ids = torch.tensor(seg_ids_l, dtype=torch.long, device=device)
    lsm = segment_logsoftmax(logits, seg_ids, seg)
    tgt = torch.tensor(targets, dtype=torch.long, device=device)
    return -lsm[tgt].mean()


def segment_logsoftmax(logits: torch.Tensor, seg_ids: torch.Tensor, n_seg: int):
    """log softmax within segments (segment ids must be >= 0 ints)."""
    maxes = torch.full((n_seg,), float("-inf"), device=logits.device)
    maxes.scatter_reduce_(0, seg_ids, logits, reduce="amax", include_self=True)
    z = logits - maxes[seg_ids]
    denom = torch.zeros(n_seg, device=logits.device)
    denom.index_add_(0, seg_ids, z.exp())
    return z - denom[seg_ids].log()

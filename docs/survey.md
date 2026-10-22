# Survey: ML-guided tensor-network contraction-order optimization

Status: initial online survey (2025). Sources cited inline.

## Problem

Given a tensor network (hypergraph: tensors = vertices, shared indices =
(hyper)edges with extents), find a pairwise contraction sequence minimizing
total work (FLOPs). NP-hard; tightly related to treewidth (Markov & Shi 2008)
and carving width (Jakes-Schauer et al., arXiv:1908.11034). We optimize FLOPs
only: no space constraint, no read-write weighting.

## Non-learned state of the art (the bar to beat)

| Method | Reference | Notes |
|---|---|---|
| Greedy / random-greedy | opt_einsum; cotengra | Fast, O(n^2 log n); weak alone |
| Hypergraph partition + greedy (KaHyPar) | cotengra | Strong single-shot |
| Hyper-optimizer (random trees + meta-param tuning) | Gray & Kourtis, Quantum 5, 410 (2021); cotengra | Strong at moderate budgets |
| SA over pairwise orders | Schindler & Jermyn, Mach. Learn.: Sci. Technol. (2020), arXiv:2001.08063 | First SA study |
| SA over contraction trees (subtree moves) | Kalachev et al., arXiv:2108.05665; TreeSA in OMEinsumContractionOrders.jl (JOSS 2026, doi:10.21105/joss.09886) | **Strongest general-purpose; TreeSA wins the TensorBFS benchmark (incl. sycamore_53_20)** |
| Exact: treewidth solvers (Tamaki, Jdrasil, FlowCutter), SAT/DP | PACE; Ibrahim et al., arXiv:2209.02895 | Small networks only |
| Cut-based / bipartition SA | SABipartite (OMEinsumContractionOrders); SEA 2024 improved cut strategy (doi:10.4230/LIPIcs.SEA.2024.27) | |
| Improved greedy cost functions | arXiv:2405.09644 | Cheap greedy upgrades |

Benchmarks to reuse: TensorBFS/OMEinsumContractionOrdersBenchmark;
Einsum Benchmark (NeurIPS 2024 Datasets & Benchmarks); QuantumBFS
quantum.harness issue #127 lists canonical hard targets (Sycamore-53 m=20,
3D lattice partition functions, hyper-optimizer benchmark graphs) with a
verification protocol (independent exact-integer cost checker).

## Learned approaches (all of them, as found)

1. **RL-TNCO** — Meirom, Maron, Mannor, Chechik (ICML 2022, arXiv:2204.09052;
   code: NVlabs/RL-TNCO). The flagship.
   - MDP: state = current weighted graph; action = pick edge to contract;
     cost = FLOPs of that contraction; minimize cumulative cost.
   - Architecture: Battaglia-style Graph Network (edge/node/global MLPs),
     edge features = {contraction cost, edge extension, opt_einsum greedy
     score} with robust log/median-split scaling; PPO with path pruning,
     optimistic buffer (off-policy replay of surprisingly good trajectories),
     greedy-prefix hybrid for large networks; inference = sample l sequences
     (50–1000), keep cheapest.
   - Results: beats OE-Greedy, CTG-Greedy, CTG-KaHyPar by 2–5x FLOPs on
     synthetic 25–100-tensor networks; competitive on Sycamore m=10..20
     (baselines given 3 h); Max-Cut 5908 tensors only via greedy prefix +
     RL suffix (1.1–3.9x).
   - **Critical limitation: never compared against TreeSA or the cotengra
     hyper-optimizer — only greedy and single-start KaHyPar.** High seed
     variance; "a lot of compute power"; RL infeasible on ~10K-edge graphs
     even on 40 GB GPUs (hence the prefix hybrid). Within-distribution
     generalization OK; downward size transfer degrades (91–99.7% retained).
2. **Learning-to-rank contraction plans** — Pastor et al. (arXiv:2608.05819).
   XGBoost ranker over 7 candidate plans per circuit (from treewidth
   heuristics: MinFill, FlowCutter, Tamaki, ...); labels = measured GPU
   wall-clock; selects, does not generate. Top-3 96% in-distribution, drops
   to 63% under family shift; 84% cross-GPU agreement. Orthogonal niche
   (plan portfolio selection), labels hardware-dependent. No cotengra/TreeSA
   baseline.
3. **RL for quantum-circuit equivalence checking** via tensor decision
   diagrams (Aalborg U., 2025): PPO+GNN again; adjacent problem.
4. **RL for ZX-calculus simplification** (NeurIPS 2024): adjacent.
5. Engineering pieces: NVIDIA patent US20230229916A1 (scalable RL TNCO);
   "Massive Parallel Simulation for TNCO Solving" (NNICE 2024) — parallel
   RL training infrastructure.

**Nothing supervised was found** (no published imitation-learning-from-
strong-oracle contraction ordering). This is a notable gap.

## Discussion: what limits progress, competing explanations

Observation: five years after RL-TNCO, no learned method holds the best
published cost on any canonical hard instance; TreeSA/SA and the
hyper-optimizer still do (quantum.harness#127). Competing explanations:

- **H1 (search-budget dominance).** Contraction ordering has a cheap, exact
  cost oracle, so simulation-free search (SA) gets enormous sample
  efficiency; learned policies trained offline cannot amortize their
  training cost unless many similar instances are solved. Cheap test:
  measure cost-vs-time curves of TreeSA on benchmark instances; if TreeSA
  saturates in seconds–minutes, the budget a learned method must beat is
  small and H1 is likely true for single-instance use.
- **H2 (representation/action space).** Edge-picking policies commit
  greedily in a space where good trees are far from any greedy prefix; the
  learned signal is not the bottleneck — the MDP factorization is. Cheap
  test: compare RL-TNCO-style per-edge policy rollouts vs. a learned
  *proposal distribution inside SA moves* on identical budgets.
- **H3 (distribution shift).** Learned methods generalize only within the
  training family (both RL-TNCO and LTR show degradation). Real value is
  cross-family transfer; no one has demonstrated it. Cheap test: train on
  random graphs, evaluate on Sycamore and vice versa.
- **H4 (weak-baseline artifact).** Published ML wins are vs. greedy/single
  KaHyPar, not vs. TreeSA at matched budget; the "learned advantage" may
  be an artifact. Cheap test: rerun RL-TNCO's own benchmark instances with
  TreeSA at matched wall-clock and compare.

Most plausible reading: H1 and H4 together — the field's learned results
are real but against the wrong bar, and the honest bar (TreeSA at matched
budget) has never been tested. H2/H3 suggest where a learned component
could still win: as guidance *inside* the strong search (proposals,
initialization, restarts, pruning) or as amortized knowledge across a
distribution of instances, not as an end-to-end replacement.

## Candidate research directions (ranked by (novelty x plausibility) / cost)

1. **Learned proposals/acceptance inside TreeSA-style SA** — a GNN scores
   candidate subtree moves; SA keeps its exact-oracle correctness, learned
   part only steers. Directly tests H2; fair-budget comparison is natural.
2. **Imitation from a strong oracle** — train a policy to predict TreeSA's
   final tree (or first moves) from the input hypergraph; use as SA
   initialization/restart policy. Supervised; no RL instability; nobody has
   published this. Tests whether structure of near-optimal trees is
   learnable at all.
3. **Amortized portfolio/verification** — predict per-instance which
   optimizer wins (cheap, likely works, low novelty).
4. **End-to-end RL improvements** (better credit assignment, etc.) —
   highest risk; H1 says the budget math is against it for single
   instances.

## Immediate next steps

1. Baseline pipeline: instance generators (random regular graphs, square
   lattices, Sycamore-family circuits) + exact FLOPs cost function +
   independent checker.
2. Protected evaluation suite: fixed instances and seeds; record
   cost-vs-time curves for TreeSA (via OMEinsumContractionOrders or a
   reimplementation) and cotengra hyper-optimizer at matched budgets.
3. Run the H1/H4 cheap tests before any learned component.

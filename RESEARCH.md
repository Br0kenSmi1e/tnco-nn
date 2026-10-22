# Research notebook

## Objective

Develop a substantively new neural-network-guided tensor-network
contraction-order algorithm that is competitive with strong existing methods
on general graph and hypergraph networks. Establish when, why, and under
what assumptions the learned component helps.

Minimize total contraction work, without a space-complexity constraint or
read-write-complexity weighting. Diagnostics, routine model variants, and
faster implementations are supporting work, not the objective.

## Comparison criteria

Evaluate best contraction work versus elapsed planning time at comparable
budgets, including training, inference, and unsuccessful search. Compare
against strong non-learned methods, including TreeSA, across relevant
networks and seeds. Separate algorithmic gains (learned guidance) from
implementation speedups and extra computation.

## Cost convention (fixed)

Primary metric: tc = log2 of total contraction work, counted as the sum
over contraction steps of the product of the extents of all indices
involved in the step (log2-sum-exp convention). No space-complexity
constraint, no read-write weighting. sc and rw are recorded as
diagnostics only.

Fairness control: TreeSA's default ScoreFunction (OMEinsumContractionOrders
v1.1.0, src/Core.jl) is tc_weight=1, sc_weight=1, rw_weight=0,
sc_target=20 — i.e., pure tc plus a penalty on peak space above 2^20,
which distorts tc search on hard instances (Sycamore m=20 has sc ~ 30).
All baselines are therefore configured with a pure-tc scorer
(ScoreFunction(tc_weight=1, sc_weight=0, rw_weight=0)). Comparing against
default TreeSA would confound algorithmic gains with its space-penalty
handicap.

Published sliced/memory-constrained numbers (Pan & Zhang Sycamore; some
cotengra benchmarks) use a different convention; cross-paper comparison
only within the unsliced tc convention.

## Starting state

Fresh project. No inherited algorithmic choices; the codebase is empty
beyond this notebook. First task is to establish a baseline pipeline and a
protected evaluation suite before any learned component is introduced.

## Resources

Use local resources reasonably. Do not incur paid external compute or change
protected evaluations without permission.

## Understanding and evidence

**Current verdict:** nothing established yet experimentally. No baselines,
no evaluation suite, no learned component.

**Survey (docs/survey.md).** The non-learned bar is TreeSA (SA over
contraction trees; OMEinsumContractionOrders.jl) and the cotengra
hyper-optimizer (Gray & Kourtis 2021); TreeSA holds the best results on
the TensorBFS benchmarks including Sycamore-53 m=20. The only published
end-to-end learned method is RL-TNCO (Meirom et al., ICML 2022): PPO +
Graph Network, action = pick edge, and it was **never compared against
TreeSA or the hyper-optimizer** — only greedy and single-start KaHyPar.
Learning-to-rank (Pastor et al., arXiv:2608.05819) selects among 7
existing plans rather than generating orders. No supervised
imitation-from-strong-oracle method exists in the literature.

**Leading hypotheses for why learned methods have not displaced SA:**
H1 search-budget dominance (exact cheap cost oracle favors SA; learned
training rarely amortizes on single instances); H4 weak-baseline artifact
(no published ML result beat TreeSA at matched budget). H2 (MDP
factorization, not learning signal, is the bottleneck) and H3 (no
cross-family transfer) point to where learning could still help: guidance
inside SA or amortized across instance distributions.

**Baseline pipeline established** (tnco/, julia/, eval/; Python 3.14 venv
via uv, Julia 1.11.5 + OMEinsumContractionOrders). Shared instance/tree
JSON format; independent exact Python checker reproduces Julia's tc/sc/rwc
exactly on all runs so far (negative control passes; sc convention = max
contraction-step output, inputs excluded). Protected suite: 6 exact-track
instances (n=20) + 14 main-track (n up to 200; random-regular, ER,
lattices; chi in {2,16}). First results: TreeSA (pure-tc scorer,
ntrials=10, niters=50) **matches ExhaustiveSearch on all six n=20
instances** (H1 confirmed at small n: no quality headroom there); greedy
is far off on hard main-track instances (e.g. ER n=100: tc 92.0 vs
TreeSA 60.6; rr n=100 chi=16: 116.0 vs 75.7). Quality headroom for a
learned method, if it exists, is at larger n where TreeSA is no longer
provably optimal.

**Next investment:** baseline pipeline + protected evaluation suite, then
the cheap H1/H4 tests (TreeSA cost-vs-time curves on benchmark instances)
before building any learned component. Candidate directions ranked in
docs/survey.md; current favorite is learned proposals inside TreeSA-style
SA or imitation from TreeSA oracles.

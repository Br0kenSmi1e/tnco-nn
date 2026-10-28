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

**Experiment 1 — TreeSA saturation (H1 test).** Escalating budget
(ntrials*niters = 500 -> 32000, 64x) on main-track instances improves tc by
< 0.1 (rr_n100_d3: 21.971 -> 21.924; rr_n200_d3: 32.062 -> 32.058;
rr_n100_d3_c16: 75.750 -> 75.743). TreeSA saturates almost immediately.
Together with exact-optimality at n=20, this supports H1: TreeSA is likely
near-optimal on these families at n <= 200; quality headroom for any method
there is small (percent-level FLOPs).

**Experiment 2 — imitation probe v1 (H2/H3 test).** EdgePolicy GNN
(hidden 128, 6 MP layers, node/edge features incl. greedy score), trained
30 epochs by teacher-forced imitation of TreeSA oracle action sequences
(371 instances, random-regular/lattice/ER mix, n<=80; oracle budget
ntrials=5, niters=100; 7/378 trees skipped for disjoint merges).
Results (eval/results/policy_v1_eval.txt): clearly beats classical greedy
(rr_n100: 22.9 vs 29.5; er_n100: 68.2 vs 92.0) but best-of-50 sampled
rollouts remain 60-460% above TreeSA in FLOPs in-distribution, worse with
size transfer (n=100), and collapse under extent transfer (chi=16: 10^7x).
Interpretation: oracle mapping is partially learnable from little data, but
per-step compounding error dominates at longer horizons. Candidate
mechanisms: (a) multimodal single-oracle targets, (b) exposure bias from
teacher forcing, (c) objective mismatch (per-step likelihood vs final tc),
(d) capacity/features. 

**Next investment:** expert-iteration / RL fine-tuning from the imitation
init, optimizing tc directly against the exact oracle (sample k rollouts,
reinforce best-of-k) — tests (b)+(c) cheaply. In parallel, extend
saturation to larger n and higher-width families to size the remaining
quality prize.

**Experiment 3 — expert iteration (EI) fine-tune**
(scripts/expert_iteration.py; 60 rounds, batch 64 instances x 16 rollouts,
best-of-k self-imitation + 0.3 oracle anchor, ~15 min on RTX 4060 Ti).
In-distribution gaps vs TreeSA roughly halved or better (rr_n60 s5000:
155% -> 28% FLOPs; rr_n80 s5001: 460% -> 82%; rr_n100 s10: 850% -> 181%),
but remain far from TreeSA, and family/extent transfer WORSENED (er_n100:
2e4% -> 7e6%; chi=16: 1e7% -> 5e12%) — self-imitation narrows the policy.
Reading: objective mismatch/exposure bias were real but partial causes;
the end-to-end learned policy path converges to 'beats greedy, does not
reach TreeSA' (RL-TNCO's story, reproduced cheaper). 

**Experiment 4 — saturation at n=500 (running):** rr_n500_d3 shows a
larger budget response than n<=200 (W=250: 71.94 -> W=1000: 71.15),
the first observed instance where TreeSA is demonstrably not saturated.
If learned-init + SA-polish beats greedy-init SA anywhere, it is at this
scale.

**Decision point:** the decisive experiment for the quality direction is
now: policy-initialized SA vs greedy-initialized SA (TreeSA default) at
matched iterations, on n=500-1000 instances. Requires custom-tree
initializer support in the Julia runner. If learned init does not help
even where TreeSA is not saturated, the quality direction closes as a
well-established negative; remaining contributions would be amortization
(equal quality, far lower planning cost) or mechanism understanding.

**Experiment 5 — saturation at scale and the init premise test.**
Saturation is scale/width-dependent (best tc by budget W=250 -> 16000):
rr_n500_d3: 71.94 -> 71.04; rr_n1000_d3: 137.1 -> 132.2 (still dropping);
er_n200: 333.6 -> 322.8; rr_n500_d5: 253.5 -> 228.4; rr_n500_d3_c16: 283.5
-> 272.7. High-width/hard instances are FAR from saturated — the quality
prize exists there (SA is mixing-limited), not at n<=200.
Premise test (rr_n500_d3, W=250/1000, seeds 11-12): random-init SA reaches
the same tc as greedy-init SA (e.g. W=1000: 71.12/71.75 random vs
71.20/71.14 greedy). On easy landscapes SA mixing is init-independent, so
learned-INIT cannot help there. Init sensitivity on hard/unsaturated
landscapes (n=1000, d5) is being tested before closing the init direction.
If init never matters, the remaining quality mechanism is learned guidance
of the SA DYNAMICS (move proposals), not the starting point.

**Experiment 6 — hybrid policy-init SA: a paradoxical negative**
(scripts/hybrid_test.py, eval/results/hybrid_n500.log). rr_n500_d3, matched
budgets, seeds 11-12. Policy best-of-50 init tree tc=104.4 (greedy init
tc=132.0; random sequential tc~200): policy-init SA reaches 76.4 (W=250)
and 73.8-75.5 (W=1000) — WORSE than greedy-init (71.5/71.2) and random-init
(72.4/71.1). A 2^27x better starting tree HURTS. Hypotheses: (a) with
initializer=:specified all ntrials start from the same tree (diversity
loss), while greedy init varies per trial; (b) policy trees are deep but
isolated optima — basin connectivity, not starting cost, determines SA
outcome. Control running: single-chain (ntrials=1) comparisons with
policy-best/policy-median/greedy inits (eval/results/hybrid_single_chain.log).
Also: rr_n500_d5 init test shows random ~ greedy (234.5/238.6 vs
241.7/238.0), i.e. no monotone init effect on rugged landscapes; only
n=1000_d3 showed random-init clearly worse (138.6/138.8 vs 134.3/131.5).

**If the single-chain control confirms (b), the learned-INIT direction is
closed with mechanism. Remaining quality mechanism: learned move PROPOSALS
in SA dynamics (target the mixing bottleneck on unsaturated landscapes),
with training data self-generated from SA traces. Cheapest mechanism probe:
hand-crafted proposal bias (e.g. cost-density-weighted subtree selection)
before any NN.**

**Experiment 7 — init paradox confirmed (6 seeds); selector mechanism
validated (the positive result).** Hybrid, W=1000, 6 seeds, rr_n500_d3:
policy-init 72.42 ± 0.4, greedy-init 71.25 ± 0.2, random-init 71.90 ± 0.9.
Init ranking by final quality is INVERTED vs init-tree cost (policy 104.4
< greedy 132.0 < random ~200). Learned-init direction CLOSED: basin
connectivity, not starting cost, determines SA outcome; low-cost sequential
policy trees occupy narrow isolated basins.

Selector probe (sa_selector.jl: TreeSA with pluggable rule selector;
random-selector mode reproduces TreeSA statistically): eps_greedy(0.3)
rule selection vs uniform-random, W=250, 3 seeds: rr_n500_d3: 71.30 vs
73.38 (eps_greedy at W=250 MATCHES random-selector at W=16000, i.e. ~64x
budget compression); er_n200: 334.2 vs 339.4 (~32x FLOPs better);
rr_n500_d5: no effect (noise). Mechanism established: on unsaturated
landscapes the proposal selector is the bottleneck; even a myopic
1-step-greedy bias helps substantially. This is the opening for a LEARNED
selector (must beat 1-step-greedy, e.g. via lookahead/global features).

**Experiment 8 — selector plateau study (rr_n500_d3, 3 seeds each):**

| selector | W=250 | W=1000 | W=4000 |
|---|---|---|---|
| random (=TreeSA) | 73.38 | 71.92 | 71.54 |
| eps_greedy(0.3) | 71.30 | 71.16 | pending |
| greedy1 | 71.19 | 71.16 | 71.04 |
| greedy2 (2-step lookahead) | **70.68** | pending | pending |

Previous best-known on this instance (random selector, W=16000): 71.04.
**greedy2 at W=250 (64x less budget) beats it by ~0.35 tc (~27% FLOPs).**
On er_n200 W=250: greedy2 331.8 < eps_greedy 334.2 < random 339.4.
Selector quality improves the PLATEAU, not just convergence speed.
Instrumented run (sa_stats.jl): fraction of visited nodes offering an
improving rule falls from ~52% (beta=0) to ~3-5% late in annealing —
most late-stage node evaluations are wasted (motivates a visit scheduler).

**Sharpened research question:** can a LEARNED selector beat greedy2?
Win conditions: (a) better plateau via learned >2-step lookahead patterns
(trainable with branch-labels: apply rule, continue K sweeps, measure);
(b) greedy2 quality at greedy1 per-sweep cost (distillation; engineering
claim, weaker). A third NN role with free supervision: visit scheduling
(predict nodes with no improving rule; skip ~95% of late-stage evals).

**Next investments:** (1) plateau study completion (greedy2 at higher W,
n=1000, er_n200 at higher W); (2) learned selector with branch-label
training; (3) visit-scheduler probe.

**Experiment 9 — affinity learnability probe (GPU-only; decides whether a
learned global-structure prior can exist).** Target: LCA cluster size of
tensor pairs in oracle trees (linearization-invariant). Model: GNN
embeddings + pair decoder; control: graph shortest-path distance. Val
Spearman (held-out instances): lattices GNN 0.79 (0.98 with random node
ids) vs control 0.37 — structure IS learnable on geometric families.
Random-regular: GNN 0.12 (0.16 with RNI) vs control 0.34 — NOT learnable
beyond graph distance; RNI ablation shows symmetry-breaking is not the
bottleneck (train MSE collapses to 0.002 = memorization; val barely
moves): the information is not present locally. ER similar (0.13 vs 0.34,
small n).
Consequence: a learned tree-structure prior is viable only for structured
(geometric) families. The instances where TreeSA is unsaturated and the
quality prize lives (er_n200, rr_n500_d5, n=1000) are exactly the
irregular ones where the prior is not learnable. Learned-global-structure
direction CLOSED for general/random networks (init paradox + this
learnability negative); remains open for circuit/lattice-like families
only.

**Surviving directions:** (1) non-learned selector improvements (greedy2
already beats best-known; worth polishing as its own result); (2) learned
rule selector / visit scheduler — needs CPU-generated SA traces (paused);
(3) premise test SABipartite vs greedy2-SA (CPU, paused).

**Open framing tension:** if saturation holds at larger n too, a pure
quality win over TreeSA may not exist on these families; the fallback
contribution is the amortization story or harder families (Sycamore-like).

**Previous note:** baseline pipeline + protected evaluation suite, then
the cheap H1/H4 tests (TreeSA cost-vs-time curves on benchmark instances)
before building any learned component. Candidate directions ranked in
docs/survey.md; current favorite is learned proposals inside TreeSA-style
SA or imitation from TreeSA oracles.

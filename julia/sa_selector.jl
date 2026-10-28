# SA over contraction trees with a PLUGGABLE rule selector.
#
# Reuses OMEinsumContractionOrders' ExprTree internals (ruleset, tcsc_diff,
# update_tree!) but replaces the uniform-random rule choice
# (`rule = rand(rst)` in optimize_subtree!) with a selector function.
# Pure-tc objective (sc_weight = 0, rw off), so dE = dtc.
#
# Usage:
#   julia --project=. sa_selector.jl INSTANCE_JSON OUT_JSON SELECTOR NTRIALS NITERS SEED [EPS]
#     SELECTOR ∈ {random, greedy1, eps_greedy}   (EPS for eps_greedy)
# Output JSON: same schema as run_optimizer.jl plus "selector".

using OMEinsumContractionOrders
const OECO = OMEinsumContractionOrders
using JSON
using Random

instance_file = ARGS[1]
out_file = ARGS[2]
selector = ARGS[3]
ntrials = parse(Int, ARGS[4])
niters = parse(Int, ARGS[5])
seed = parse(Int, ARGS[6])
eps = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 0.1
βs = collect(0.01:0.05:15)  # TreeSA default schedule

Random.seed!(seed)

inst = JSON.parsefile(instance_file)
inputs = [Vector{Int}(ix) for ix in inst["inputs"]]
output = Vector{Int}(inst["output"])
size_dict = Dict(parse(Int, k) => v for (k, v) in inst["size_dict"])
code = OECO.EinCode(inputs, output)

labels = OECO._label_dict(code)
inverse_map = Dict([v => k for (k, v) in labels])
log2_sizes = [log2(size_dict[inverse_map[i]]) for i in 1:length(labels)]

function select_rule(dEs)
    if selector == "random"
        return rand(1:length(dEs))
    elseif selector == "greedy1"
        return argmin(dEs)
    elseif selector == "eps_greedy"
        return rand() < eps ? rand(1:length(dEs)) : argmin(dEs)
    else
        error("unknown selector $selector")
    end
end

# ---- 2-step lookahead support (selector == "greedy2") ----
# Snapshot/restore of the 2-level neighborhood a rule can mutate.
function snapshot_nbhd(tree)
    L, R = tree.left, tree.right
    lsnap = OECO.isleaf(L) ? nothing : (L.left, L.right, L.info)
    rsnap = OECO.isleaf(R) ? nothing : (R.left, R.right, R.info)
    return (L, R, lsnap, rsnap)
end
function restore_nbhd!(tree, snap)
    L, R, lsnap, rsnap = snap
    tree.left, tree.right = L, R
    if lsnap !== nothing
        L.left, L.right, L.info = lsnap
    end
    if rsnap !== nothing
        R.left, R.right, R.info = rsnap
    end
    return tree
end

# best immediate dtc available at a child subtree (0 if no rule applies)
function best_child_dtc(sub, log2_sizes)
    rst = OECO.ruleset(OECO.TreeDecomp(), sub)
    isempty(rst) && return 0.0
    return minimum(d[2] - d[1] for d in
                   (OECO.tcsc_diff(sub, r, log2_sizes, false) for r in rst))
end

function select_rule_lookahead(tree, rst, diffs, dEs, log2_sizes)
    best_i, best_score = 1, Inf
    for i in 1:length(rst)
        snap = snapshot_nbhd(tree)
        OECO.update_tree!(tree, rst[i], diffs[i][6])
        future = min(best_child_dtc(tree.left, log2_sizes),
                     best_child_dtc(tree.right, log2_sizes))
        restore_nbhd!(tree, snap)
        score = dEs[i] + min(0.0, future)
        if score < best_score
            best_score, best_i = score, i
        end
    end
    return best_i
end

function sa_sweep!(tree, β, log2_sizes)
    rst = OECO.ruleset(OECO.TreeDecomp(), tree)
    if !isempty(rst)
        # evaluate all applicable rules (selector needs their deltas)
        diffs = [OECO.tcsc_diff(tree, r, log2_sizes, false) for r in rst]
        dEs = [d[2] - d[1] for d in diffs]
        i = if selector == "greedy2"
            select_rule_lookahead(tree, rst, diffs, dEs, log2_sizes)
        else
            select_rule(dEs)
        end
        if rand() < exp(-β * dEs[i])  # Metropolis
            OECO.update_tree!(tree, rst[i], diffs[i][6])
        end
        for subtree in OECO.siblings(tree)
            sa_sweep!(subtree, β, log2_sizes)
        end
    end
end

function main()
    t0 = time()
    best_tree = nothing
    best_tc = Inf
    for t in 1:ntrials
        tree = OECO._initializetree(code, size_dict, :greedy, OECO.TreeDecomp())
        for β in βs
            for _ in 1:niters
                sa_sweep!(tree, β, log2_sizes)
            end
        end
        tc, sc, rw = OECO.tree_timespace_complexity(tree, log2_sizes)
        if tc < best_tc
            best_tc = tc
            best_tree = tree
        end
    end
    elapsed = time() - t0

    ne = OECO.NestedEinsum(best_tree, inverse_map)
    cc = contraction_complexity(ne, size_dict)

    function to_tree(ne)
        OECO.isleaf(ne) && return ne.tensorindex - 1
        return [to_tree(arg) for arg in ne.args]
    end

    result = Dict(
        "tc" => cc.tc, "sc" => cc.sc, "rwc" => cc.rwc, "tree" => to_tree(ne),
        "optimizer" => "sa_selector", "selector" => selector, "seed" => seed,
        "ntrials" => ntrials, "niters" => niters, "elapsed_seconds" => elapsed,
    )
    open(out_file, "w") do f
        JSON.print(f, result)
    end
    println(JSON.json(Dict("tc" => cc.tc, "sc" => cc.sc, "elapsed" => elapsed)))
end

main()

# Run a contraction-order optimizer on a tensor-network instance.
#
# Usage:
#   julia --project=. run_optimizer.jl INSTANCE_JSON OUT_JSON OPTIMIZER [SEED] [NTRIALS] [NITERS]
#
#   OPTIMIZER ∈ {treesa, greedy, exact}
#     treesa : TreeSA with pure-tc scorer (ntrials, niters apply)
#     greedy : GreedyMethod
#     exact  : ExhaustiveSearch (small networks only)
#   Optional ARGS[7]: TreeSA initializer (:greedy, :random, :specified)
#   Optional ARGS[8]: path to init-tree JSON (nested list, 0-based leaves);
#                     forces initializer=:specified
#
# Instance JSON format (shared with the Python side):
#   {"inputs": [[1,2],[2,3],...],   # tensor i has indices inputs[i]
#    "output": [],                  # open indices (usually empty)
#    "size_dict": {"1": 2, ...}}    # index label (as string) -> extent
#
# Output JSON:
#   {"tc": ..., "sc": ..., "rwc": ...,       # log2 complexities (Julia-computed)
#    "tree": nested list, leaf = tensor index (0-based Int), node = [left, right],
#    "optimizer": ..., "seed": ..., "ntrials": ..., "niters": ...}
#
# The Python checker recomputes tc/sc/rwc from "tree" independently.

using OMEinsumContractionOrders
const OECO = OMEinsumContractionOrders
using JSON
using Random

instance_file = ARGS[1]
out_file = ARGS[2]
opt_name = ARGS[3]
seed    = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 42
ntrials = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 10
niters  = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 50
initializer = length(ARGS) >= 7 ? Symbol(ARGS[7]) : :greedy
init_tree_file = length(ARGS) >= 8 ? ARGS[8] : nothing

Random.seed!(seed)

inst = JSON.parsefile(instance_file)
inputs = [Vector{Int}(ix) for ix in inst["inputs"]]
output = Vector{Int}(inst["output"])
size_dict = Dict(parse(Int, k) => v for (k, v) in inst["size_dict"])

# Build a NestedEinsum from the shared nested-list tree format.
# Returns (NestedEinsum, live index set of the subtree, leaf set).
function build_nested(tree, inputs, output_set, covered)
    if tree isa Integer
        idx = tree + 1  # shared format is 0-based
        return OECO.NestedEinsum{Int}(idx), Set(inputs[idx]), Set([idx])
    end
    left, sl, ll = build_nested(tree[1], inputs, output_set, covered)
    right, sr, lr = build_nested(tree[2], inputs, output_set, covered)
    leaves = union(ll, lr)
    live = Set{Int}()
    for l in union(sl, sr)
        if l in output_set || !(covered[l] <= leaves)
            push!(live, l)
        end
    end
    eins = OECO.EinCode([collect(sl), collect(sr)], collect(live))
    return OECO.NestedEinsum(OECO.NestedEinsum{Int}[left, right], eins), live, leaves
end

code = OECO.EinCode(inputs, output)

optimizer = if opt_name == "treesa"
    # Pure-tc scorer: no space penalty, no read-write weighting.
    if init_tree_file !== nothing
        initializer = :specified
    end
    TreeSA(; ntrials=ntrials, niters=niters, initializer=initializer,
           score=ScoreFunction(tc_weight=1.0, sc_weight=0.0, rw_weight=0.0))
elseif opt_name == "greedy"
    GreedyMethod()
elseif opt_name == "exact"
    ExhaustiveSearch()
else
    error("unknown optimizer `$opt_name`")
end

t0 = time()
ne = if init_tree_file !== nothing && opt_name == "treesa"
    init_tree = JSON.parsefile(init_tree_file)
    covered = Dict{Int,Set{Int}}()
    for (t, ixs) in enumerate(inputs), ix in ixs
        push!(get!(covered, ix, Set{Int}()), t)
    end
    ne0, _, _ = build_nested(init_tree, inputs, Set(output), covered)
    optimize_code(ne0, size_dict, optimizer)
else
    optimize_code(code, size_dict, optimizer)
end
elapsed = time() - t0
cc = contraction_complexity(ne, size_dict)

# Convert NestedEinsum to the shared nested-list tree format (0-based).
function to_tree(ne)
    if OECO.isleaf(ne)
        return ne.tensorindex - 1
    else
        return [to_tree(arg) for arg in ne.args]
    end
end

result = Dict(
    "tc" => cc.tc, "sc" => cc.sc, "rwc" => cc.rwc,
    "tree" => to_tree(ne),
    "optimizer" => opt_name, "seed" => seed, "initializer" => String(initializer),
    "ntrials" => ntrials, "niters" => niters,
    "elapsed_seconds" => elapsed,
)
open(out_file, "w") do f
    JSON.print(f, result)
end
println(JSON.json(Dict("tc" => cc.tc, "sc" => cc.sc, "rwc" => cc.rwc,
                       "elapsed_seconds" => elapsed)))

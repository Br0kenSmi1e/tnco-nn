# Run a contraction-order optimizer on a tensor-network instance.
#
# Usage:
#   julia --project=. run_optimizer.jl INSTANCE_JSON OUT_JSON OPTIMIZER [SEED] [NTRIALS] [NITERS]
#
#   OPTIMIZER ∈ {treesa, greedy, exact}
#     treesa : TreeSA with pure-tc scorer (ntrials, niters apply)
#     greedy : GreedyMethod
#     exact  : ExhaustiveSearch (small networks only)
#   Optional ARGS[7]: TreeSA initializer (:greedy or :random)
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

Random.seed!(seed)

inst = JSON.parsefile(instance_file)
inputs = [Vector{Int}(ix) for ix in inst["inputs"]]
output = Vector{Int}(inst["output"])
size_dict = Dict(parse(Int, k) => v for (k, v) in inst["size_dict"])

code = OECO.EinCode(inputs, output)

optimizer = if opt_name == "treesa"
    # Pure-tc scorer: no space penalty, no read-write weighting.
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
ne = optimize_code(code, size_dict, optimizer)
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

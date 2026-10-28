# Statistics probe: during random-selector SA, what fraction of visited
# internal nodes have at least one improving rule, as a function of
# annealing progress (beta)? Motivates a learned visit scheduler.
#
# Usage: julia --project=. sa_stats.jl INSTANCE_JSON NTRIALS NITERS SEED

using OMEinsumContractionOrders
const OECO = OMEinsumContractionOrders
using JSON
using Random

instance_file = ARGS[1]
ntrials = parse(Int, ARGS[2])
niters = parse(Int, ARGS[3])
seed = parse(Int, ARGS[4])
βs = collect(0.01:0.05:15)

Random.seed!(seed)
inst = JSON.parsefile(instance_file)
inputs = [Vector{Int}(ix) for ix in inst["inputs"]]
output = Vector{Int}(inst["output"])
size_dict = Dict(parse(Int, k) => v for (k, v) in inst["size_dict"])
code = OECO.EinCode(inputs, output)
labels = OECO._label_dict(code)
inverse_map = Dict([v => k for (k, v) in labels])
log2_sizes = [log2(size_dict[inverse_map[i]]) for i in 1:length(labels)]

function sweep_stats!(tree, β, log2_sizes)
    nvisited = 0
    nimproving = 0
    function rec(t)
        rst = OECO.ruleset(OECO.TreeDecomp(), t)
        if !isempty(rst)
            nvisited += 1
            dEs = [d[2] - d[1] for d in (OECO.tcsc_diff(t, r, log2_sizes, false) for r in rst)]
            if minimum(dEs) < 0
                nimproving += 1
            end
            i = rand(1:length(rst))  # random selector
            if rand() < exp(-β * dEs[i])
                OECO.update_tree!(t, rst[i], OECO.tcsc_diff(t, rst[i], log2_sizes, false)[6])
            end
            for sub in OECO.siblings(t)
                rec(sub)
            end
        end
    end
    rec(tree)
    return nvisited, nimproving
end

function main()
    tree = OECO._initializetree(code, size_dict, :greedy, OECO.TreeDecomp())
    nb = length(βs)
    for (bi, β) in enumerate(βs)
        vis = imp = 0
        for _ in 1:niters
            v, im = sweep_stats!(tree, β, log2_sizes)
            vis += v; imp += im
        end
        if bi % 30 == 1 || bi == nb
            tc = OECO.tree_timespace_complexity(tree, log2_sizes)[1]
            println("β=$(round(β, digits=2)) frac_improving=$(round(imp / max(vis, 1), digits=4)) tc=$(round(tc, digits=3))")
            flush(stdout)
        end
    end
end

main()

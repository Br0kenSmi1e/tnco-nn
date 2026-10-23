# Generate TreeSA oracle trees for a shard of training instances.
#
# Usage: julia --project=. gen_oracle.jl JOBS_JSONL
#   each line: {"instance": path, "out": path, "ntrials": N, "niters": M, "seed": S}
#   writes result JSON (same format as run_optimizer.jl) to "out".
# Skips jobs whose output already exists. Prints one line per job.

using OMEinsumContractionOrders
const OECO = OMEinsumContractionOrders
using JSON
using Random

function to_tree(ne)
    if OECO.isleaf(ne)
        return ne.tensorindex - 1
    else
        return [to_tree(arg) for arg in ne.args]
    end
end

for line in eachline(ARGS[1])
    job = JSON.parse(line)
    if isfile(job["out"])
        println("skip ", job["out"])
        continue
    end
    Random.seed!(job["seed"])
    inst = JSON.parsefile(job["instance"])
    inputs = [Vector{Int}(ix) for ix in inst["inputs"]]
    output = Vector{Int}(inst["output"])
    size_dict = Dict(parse(Int, k) => v for (k, v) in inst["size_dict"])
    code = OECO.EinCode(inputs, output)
    score = ScoreFunction(tc_weight=1.0, sc_weight=0.0, rw_weight=0.0)
    optimizer = TreeSA(; ntrials=job["ntrials"], niters=job["niters"], score=score)
    t0 = time()
    ne = optimize_code(code, size_dict, optimizer)
    cc = contraction_complexity(ne, size_dict)
    result = Dict(
        "tc" => cc.tc, "sc" => cc.sc, "rwc" => cc.rwc, "tree" => to_tree(ne),
        "optimizer" => "treesa", "seed" => job["seed"],
        "ntrials" => job["ntrials"], "niters" => job["niters"],
        "elapsed_seconds" => time() - t0,
    )
    open(job["out"], "w") do f
        JSON.print(f, result)
    end
    println("done ", job["instance"], " tc=", round(cc.tc, digits=4), " t=", round(time() - t0, digits=1), "s")
    flush(stdout)
end

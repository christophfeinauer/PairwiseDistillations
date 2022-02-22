using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using PdbTool
using ArgParse
using HDF5
using LinearAlgebra: norm
using Statistics
using UnicodePlots
using JLD
using ArDCA

function zero_sum_gauge!(couplings)
    q = size(couplings, 2)
    @assert q == size(couplings, 3)
    for l in 1:size(couplings, 1)
        cmat = copy(couplings[l, :, :])
        cmat -= repeat(mean(cmat, dims=1), q, 1)
        cmat -= repeat(mean(cmat, dims=2), 1, q)
        cmat .+= mean(cmat)
        couplings[l, :, :] = cmat[:, :]
    end
end

function create_hmm(fasta_path)
    hmm_path = tempname()
    run(pipeline(fasta_path, `sed '/^>/!s/[a-z,\.]//g'`, `hmmbuild -n hmm --symfrac=0.0 --informat=a2m $hmm_path - `))
    return hmm_path
end

function get_predictions(model_path)

    couplings = h5read(model_path, "couplings")
    fields = h5read(model_path, "fields")

    zero_sum_gauge!(couplings)

    N = size(fields, 1)

    F = zeros(N,N)
    l = 1
    for i in 1:N
        for j in (i+1):N
            F[i,j] = norm(couplings[l ,1:end-1, 1:end-1])
            F[j,i] = F[i,j]
            l += 1
        end
    end
    @assert l == size(couplings,1)+1

    Fi = sum(F, dims=1)
    Fj = sum(F, dims=2)
    Fa = sum(F) * (1-1/N)
    F -= (Fj * Fi) / Fa

    preds = Vector{Tuple{Int64, Int64, Float64}}()
    for i in 1:N
        for j in (i+1):N
            push!(preds, (i,j,F[i,j]))
        end
    end

    print(heatmap(F))

    sort!(preds, by=k->k[3], rev=true)
    println(preds[1:10])

    return preds
end

function get_roc(fasta_path, model_path, pdb_path, chain)

    pdb = PdbTool.parsePdb(pdb_path)
    hmm_path = create_hmm(fasta_path)
    PdbTool.mapChainToHmm(pdb.chain[chain], hmm_path)
    if occursin("ardca", model_path) && !occursin("extracted", model_path)
        preds = load(model_path)["episcore"]
    else
        preds = get_predictions(model_path)
    end
    roc, intraAlignDist = PdbTool.makeIntraRoc(preds, pdb.chain[chain]), PdbTool.intraAlignDist(pdb.chain[chain])

    # replace pdb ids with align ids
    align_roc = []
    for i in 1:length(roc)
        a1 = pdb.chain[chain].residue[roc[i][1]].alignmentPos
        a2 = pdb.chain[chain].residue[roc[i][2]].alignmentPos
        r = roc[i][3]
        push!(align_roc, (a1, a2, r))
    end

    return align_roc, intraAlignDist

end


s = ArgParseSettings()

@add_arg_table s begin
    "--fasta_path"
    required = true
    "--model_path"
    required = true
    "--pdb_path"
    required = true
    "--chain"
    default="A"
    "--roc_out_file"
    required = true
    "--cmap_out_file"
end

parsed_args = parse_args(ARGS, s)

roc, intraAlignDist = get_roc(parsed_args["fasta_path"], parsed_args["model_path"], parsed_args["pdb_path"], parsed_args["chain"])

open(parsed_args["roc_out_file"], "w") do fid
    for i in 1:length(roc)
        println(fid, roc[i][1], ' ', roc[i][2], ' ', roc[i][3])
    end
end

if "cmap_out_file" in keys(parsed_args)
        open(parsed_args["cmap_out_file"], "w") do fid
        N = size(intraAlignDist, 1)
        for i in 1:N
            for j in (i+1):N
                println(fid, i, ' ', j, ' ', intraAlignDist[i,j])
            end
        end
    end
end

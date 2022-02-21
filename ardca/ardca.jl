using Distributed
using ArDCA
using ArDCA: _outputarnet!
using JLD
using ExtractMacro: @extract
using LinearAlgebra: norm
using Base.Threads: @threads, nthreads, threadid
using ProgressMeter
using UnicodePlots
using PdbTool
using LoopVectorization
using Statistics
using ArgParse
using DCAUtils
using Printf
using HDF5
using Random


function make_chunks(M, chunk_size)
    return [i:min(M, i+chunk_size-1) for i in 1:chunk_size:M]
end

function make_pairs(N)
    npairs = binomial(N,2)
    l = 0
    pairs = []
    for i in 1:N
        for j in (i+1):N
            l+=1
            push!(pairs, (i,j,l))
        end
    end
end

function load_model(model_path)

    if !isfile(model_path)
        error("cannot load $model_path")
    end
    data = load(model_path)
    arnet = data["arnet"]
    episcore = data["episcore"]
    return arnet, episcore

end

function train_model(fasta_path, model_path, overwrite, randomize; lambdaJ=0.01, lambdaH=0.0001, theta=0.2, permorder=:NATURAL, max_gap_fraction=1)

    if isfile(model_path) && !overwrite
        error("Model $model_path exists")
    end


    @show fasta_path
    arnet, arvar = ardca(fasta_path, verbose=true, lambdaJ=lambdaJ, lambdaH=lambdaH, theta=theta, permorder=permorder, max_gap_fraction=max_gap_fraction)
    if randomize
        for _H in arnet.H
            _H[:] = randn(length(_H))
        end
        for _J in arnet.J
            _J[:] = randn(length(_J))
        end
    end
    episcore = epistatic_score(arnet, arvar, 1)
    save(model_path, "arnet", arnet, "episcore", episcore)

    return arnet, episcore

end


function getNq(arnet::ArNet)
    N = length(arnet.idxperm)
    q = length(arnet.H[1])
    return N, q
end

function get_plike(x, J, H, p0, N, q)
    plike = log(p0[x[1]])
    totH = zeros(q)
    for site in 1:N-1
        Js = J[site]
        h = H[site]
        copy!(totH,h)
        @avx for i in 1:site
            for a in 1:q
                totH[a] += Js[a,x[i],i]
            end
        end
        ArDCA.softmax!(totH)
        plike += log(totH[x[site+1]])
    end
    return plike
end


function extract_couplings_q0(arnet, gauge_aa; pc=0.01)


    idxperm = arnet.idxperm
    H = arnet.H
    J = arnet.J
    p0 = arnet.p0

    N, q = getNq(arnet)
    L = binomial(N,2)

    total = L * q * q + N * q + 1

    backorder = sortperm(arnet.idxperm)
    plike_mean = 0
    R = zeros(L, q, q)
    Q = zeros(N, q)

    progress = Progress(total)

    ppc = (1-pc) * arnet.p0 + pc * ones(q)/q

    # get gauge
    seq = fill(gauge_aa, N)
    gauge = get_plike(seq[idxperm], J, H, ppc, N, q)
    next!(progress)

    for i in 1:N
        for a in 1:q
            a == gauge_aa && continue
            seq = fill(gauge_aa, N)
            seq[i] = a
            Q[i,a] = get_plike(seq[idxperm], J, H, ppc, N, q) - gauge
            next!(progress)
        end
    end


    l = 0
    for i in 1:N
        for j in (i+1):N
            l += 1
            for a in 1:q
                for b in 1:q
                    b == gauge_aa && continue
                    seq = fill(gauge_aa, N)
                    seq[i] = a
                    seq[j] = b
                    R[l,a,b] = get_plike(seq[idxperm], J, H, ppc, N, q) - Q[i,a] - Q[j,b] - gauge
                    next!(progress)
                end
            end
        end
    end


    return Q, R

end

function extract_couplings_thread!(id, arnet, ppc, R, Q, R_count, Q_count, M_thread)

    @extract arnet : H J p0 idxperm

    N, q = getNq(arnet)
    Z = rand(1:q, N, M_thread)

    plikes = zeros(M_thread)
    for m in 1:M_thread
        plikes[m] = get_plike(Z[:, m], J, H, ppc, N, q)
    end

    ArDCA.permuterow!(Z, sortperm(idxperm))

    for i in 1:N
        @avx for m in 1:M_thread
            plike = plikes[m]
            a = Z[i,m]
            Q[i, a] += plike
            Q_count[i, a] += 1
        end
    end

    l = 0
    for i in 1:N
        for j in (i+1):N
            l += 1
            @avx for m in 1:M_thread
                plike = plikes[m]
                a = Z[i,m]
                b = Z[j,m]
                R[l,a,b] += plike
                R_count[l,a,b] += 1
            end
        end
    end


    return sum(plikes)

end


function extract_couplings(arnet, M; pc=0.01, chunks_per_thread=1000)


    idxperm = arnet.idxperm
    H = arnet.H
    J = arnet.J
    p0 = arnet.p0

    N, q = getNq(arnet)
    L = binomial(N,2)

    nt = nthreads()

    backorder = sortperm(arnet.idxperm)
    plike_mean = 0
    R = zeros(nt, L, q, q)
    Q = zeros(nt, N, q)
    R_count = zeros(nt, L, q, q)
    Q_count = zeros(nt, N, q)

    chunk_size_M = cld(M, nthreads()*chunks_per_thread)
    chunks_M = make_chunks(M, chunk_size_M)
    progress = Progress(length(chunks_M), 1)

    ppc = (1-pc) * arnet.p0 + pc * ones(q)/q

    plikes = zeros(nt)
    plikes_count = zeros(nt)

    @threads for chunk in chunks_M

       id = threadid()

       R_thread = @view R[id, :, :, :]
       Q_thread = @view Q[id, :, :]
       R_count_thread = @view R_count[id, :, :, :]
       Q_count_thread = @view Q_count[id, :, :]


       m = extract_couplings_thread!(id,
                                     arnet,
                                     ppc,
                                     R_thread,
                                     Q_thread,
                                     R_count_thread,
                                     Q_count_thread,
                                     length(chunk))

       plikes[id] += m
       plikes_count[id] += length(chunk)
       next!(progress)

    end


    plike_mean = sum(plikes)/sum(plikes_count)

    R = dropdims(sum(R, dims=1), dims=1)
    R_count = dropdims(sum(R_count, dims=1), dims=1)
    Q = dropdims(sum(Q, dims=1), dims=1)
    Q_count = dropdims(sum(Q_count, dims=1), dims=1)

    R[:] = R[:]./R_count[:]
    Q[:] = Q[:]./Q_count[:] .- plike_mean
    l = 0
    for i in 1:N
        for j in (i+1):N
            l+=1
            for a in 1:q
                for b in 1:q
                    R[l, a, b] -= (Q[i, a] + Q[j,b] + plike_mean)
                end
            end
        end
    end
    return Q, R
end

function extract_or_load_couplings(arnet, model_path, M)

    coupling_file = model_path*".$M.jld"
    if isfile(coupling_file)
        println("Loading couplings from $coupling_file")
        data = load(coupling_file)
        Q = data["Q"]
        R = data["R"]
        return Q, R
    else
        Q, R = extract_couplings(arnet; M=M)
        save(coupling_file, "Q", Q, "R", R)
        return Q, R
    end
end

function frobnorm(R, N)
    L = binomial(N,2)
    F = zeros(N, N)
    l = 0
    for i in 1:N
        for j in (i+1):N
            l += 1
            #f = norm(R[l, 1:end-1, 1:end-1])
            f = norm(R[l, :, :])
            F[i,j] = f
            F[j,i] = f
        end 
    end
    return F
end

function get_num2letter(q)
    num2letter = fill('-', q)
    for c in 'A':'Z'
        num = DCAUtils.ReadFastaAlignment.letter2num(c)
        if num == q
            continue
        end
        num2letter[num] = c
    end
    return num2letter
end

function get_plike!(Z, q, arnet, pc)
    N, M = size(Z)
    ppc = (1-pc) * arnet.p0 + pc * ones(q)/q
    ArDCA.permuterow!(Z, arnet.idxperm)
    plikes = [get_plike(Z[:,m], arnet.J, arnet.H, ppc, N, q) for m in 1:M]
    return plikes
end

function get_plike(Z, q, arnet, pc)
    plikes = get_plike!(Z, q, arnet, pc)
    ArDCA.permuterow!(Z, sortperm(arnet.idxperm))
    return plikes
end


function sample_arnet(arnet, N, q, pc, nsamples, sample_batch_size, out_file, sample_dist="M")
    if nsamples % sample_batch_size != 0
        error("sample_batch_size has to divide nsamples")
    end
    num2letter = get_num2letter(q)
    samples = zeros(Int8, nsamples, N)
    logp = zeros(Float64, nsamples)
    nbatches = div(nsamples, sample_batch_size)
    progress = Progress(nbatches)
    sample_id = 1
    for batch_id in 1:nbatches
        if sample_dist=="M"
            samples_batch = sample(arnet, sample_batch_size)
        else
            samples_batch = rand(1:q, N, sample_batch_size)
        end
        logp_batch = get_plike(samples_batch, q, arnet, pc)
        for m_batch in 1:sample_batch_size
            logp[sample_id] = logp_batch[m_batch]
            for pos in 1:N
                samples[sample_id, pos] = samples_batch[pos, m_batch]
            end
            sample_id += 1
        end
        next!(progress)
    end
    @assert sample_id == nsamples+1
    # need to transpose and subtract 1
    h5write(out_file, "samples", convert.(Int8, transpose(samples) .- 1))
    h5write(out_file, "logp", convert(Array{Float32}, logp))
end

args = ArgParseSettings()
@add_arg_table! args begin
    "--gauge_aa"
    arg_type=Int
    default=21
    "--model_path"
    default=nothing
    "--fasta_path"
    default=nothing
    "--out_file"
    default="stdout"
    "--overwrite"
    action=:store_true
     "--extracted_couplings_path"
    default=nothing
    "--randomize"
    action=:store_true
    "--M"
    arg_type=Int
    default=nothing
    "--pc"
    arg_type=Float64
    default=0.1
    "--lambdaJ"
    arg_type=Float64
    default=0.01
    "--lambdaH"
    arg_type=Float64
    default=0.0001
    "--theta"
    arg_type=Float64
    default=0.2
    "mode"
    help="either train, extract, evaluate or randomize"
    required=true
    "--sample_batch_size"
    arg_type=Int
    default=1000
    "--nsamples"
    arg_type=Int
    default=10000
    "--sample_dist"
    arg_type=String
    "--chain"
    default="A"
    "--pdb_path"
    default=nothing
    "--seed"
    default=1
end


parsed_args = parse_args(ARGS, args)

# first of all we seed the RNG
Random.seed!(parsed_args["seed"])


# train
if parsed_args["mode"]=="train"
    if !("fasta_path" in keys(parsed_args))
        error("fasta_path not set")
    end
    if !("model_path" in keys(parsed_args))
        error("model_path not set")
    end
    train_model(parsed_args["fasta_path"],
                parsed_args["model_path"],
                parsed_args["overwrite"],
                parsed_args["randomize"];
                lambdaH = parsed_args["lambdaH"],
                lambdaJ = parsed_args["lambdaJ"])
end

# extract
if parsed_args["mode"]=="extract"
    if !("model_path" in keys(parsed_args))
        error("model_path not set")
    end
    if !("extracted_couplings_path" in keys(parsed_args))
        error("extracted_couplings_path not set")
    end
    if !("M" in keys(parsed_args))
        error("M not set")
    end

    arnet, episcore = load_model(parsed_args["model_path"])
    Q, R = extract_couplings(arnet, parsed_args["M"])
    h5open(parsed_args["extracted_couplings_path"], "w") do fid
        write(fid, "fields", Q)
        write(fid, "couplings", R)
    end
end

# extract q0
if parsed_args["mode"]=="extract_q0"
    if !("model_path" in keys(parsed_args))
        error("model_path not set")
    end
    if !("extracted_couplings_path" in keys(parsed_args))
        error("extracted_couplings_path not set")
    end
    if !("gauge_aa" in keys(parsed_args))
        error("gauge_aa not set")
    end

    arnet, episcore = load_model(parsed_args["model_path"])
    Q, R = extract_couplings_q0(arnet, parsed_args["gauge_aa"])
    h5open(parsed_args["extracted_couplings_path"], "w") do fid
        write(fid, "fields", Q)
        write(fid, "couplings", R)
    end
end

if parsed_args["mode"]=="evaluate"
    if !("model_path" in keys(parsed_args))
        error("model_path not set")
    end
    if !("fasta_path" in keys(parsed_args))
        error("fasta_path not set")
    end
    if !("out_file" in keys(parsed_args))
        error("out_file not set")
    end

    arnet, episcore = load_model(parsed_args["model_path"])
    N, q = getNq(arnet)
    pc = parsed_args["pc"]
    Z = DCAUtils.read_fasta_alignment(parsed_args["fasta_path"], 1.0)
    ArDCA.permuterow!(Z, arnet.idxperm)
    @assert N == size(Z,1)
    M = size(Z,2)

    ppc = (1-pc) * arnet.p0 + pc * ones(q)/q
    plikes = [get_plike(Z[:,m], arnet.J, arnet.H, ppc, N, q) for m in 1:M]
    out_file = parsed_args["out_file"]
    if parsed_args["out_file"]=="stdout"
        for m in 1:length(plikes)
            @printf("%f\n", plikes[m])
        end
    else
        # attention: need to revert permutation
        backorder = sortperm(arnet.idxperm)
        ArDCA.permuterow!(Z, backorder)
        h5write(out_file, "samples", convert.(Int8, Z .- 1))
        h5write(out_file, "logp", convert(Array{Float32}, plikes))
   end
end

function get_ppv(episcore, pdb_path, fasta_path, chain)
    hmm_path = tempname()
    run(pipeline(fasta_path, `sed '/^>/!s/[a-z,\.]//g'`, `hmmbuild -n hmm --symfrac=0.0 --informat=a2m $hmm_path - `))
    pdb = PdbTool.parsePdb(pdb_path)
    PdbTool.mapChainToHmm(pdb.chain[chain], hmm_path)
    roc = PdbTool.makeIntraRoc(episcore, pdb.chain[chain]) 
    println(lineplot([roc[i][3] for i in 1:100]))
end


if parsed_args["mode"]=="randomize"
    if !("model_path" in keys(parsed_args))
        error("model_path not set")
    end
    if !("out_file" in keys(parsed_args))
        error("out_file not set")
    end
    arnet, episcore = load_model(parsed_args["model_path"])
    N, q = getNq(arnet)
    for subJ in arnet.J
        subJ[:] = randn(size(subJ))[:]
    end
    for subH in arnet.H
        subH[:] = randn(size(subH))
    end
    arnet.idxperm[:] = randperm(N)[:]
    save(parsed_args["out_file"]*".jld", "arnet", arnet, "episcore", episcore)
end

if parsed_args["mode"]=="sample"
    if !("model_path" in keys(parsed_args))
        error("model_path not set")
    end
    if !("out_file" in keys(parsed_args))
        error("out_file not set")
    end
    if !("nsamples" in keys(parsed_args))
        error("nsamples not set")
    end
    arnet, episcore = load_model(parsed_args["model_path"])
    N, q = getNq(arnet)
    samples = sample_arnet(arnet, N, q, parsed_args["pc"], parsed_args["nsamples"], parsed_args["sample_batch_size"], parsed_args["out_file"], parsed_args["sample_dist"])
end

if parsed_args["mode"]=="ppv"
    if !("model_path" in keys(parsed_args))
        error("model_path not set")
    end
    if !("pdb_path" in keys(parsed_args))
        error("pdb_path not set")
    end
    if !("fasta_path" in keys(parsed_args))
        error("fasta_path not set")
    end
    arnet, episcore = load_model(parsed_args["model_path"])
    get_ppv(episcore, parsed_args["pdb_path"], parsed_args["fasta_path"], parsed_args["chain"])
end




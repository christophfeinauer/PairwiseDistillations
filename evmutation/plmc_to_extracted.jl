using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using HDF5

# the file layout coming from plmc is
# nsites            int32   (1)
# ncodes            int32   (1)
# nseqs             int32   (1)
# nskippedseqs      int32   (1)
# niter             int32   (1)
# theta             cfloat  (1) 
# lh                cfloat  (1)
# le                cfloat  (1)
# lg                cfloat  (1)
# neff              cfloat  (1)
# alphabet          int8    (ncodes)
# ??                cfloat  (nseqs + nskippedseqs)
# focus seq         int8    (nsites)
# offset map        int32   (nsites)
# 1pmarginals       cfloat  (nsites*ncodes)
# fields            cfloat  (nsites*ncodes)
# 2pmarginals       cfloat  (choose(sites,2) * ncodes * ncodes)
# couplings         cfloat  (choose(sites,2) * ncodes * ncodes

function parse_plmc(plmc_file, out_file)

    # open file
    fid = open(plmc_file)

    # get nsites, ncodes, nseqs and nskippedseqs
    buffer = zeros(UInt8, 4*sizeof(Cint))
    readbytes!(fid, buffer)
    nsites, ncodes, nseqs, nskippedseqs = Int.(reinterpret(Cint, buffer))

    @show nsites, ncodes, nseqs, nskippedseqs

    # skip the niter, theta, lh, le, lg, neff, ??? and focus_seq
    to_skip = sizeof(Cint)*(1+nsites) + sizeof(UInt8)*(ncodes+nsites) +  sizeof(Cfloat)*(5 + nseqs + nskippedseqs)
    skip(fid, to_skip)

    # get 1pmarginals (we only use them as consistency check)
    nfields = nsites * ncodes
    buffer = zeros(UInt8, sizeof(Cfloat)*nfields)
    readbytes!(fid, buffer)

    # the sum of the 1pmarginals should be equal to nsites
    @assert abs(sum(reinterpret(Cfloat, buffer)) - nsites) < 0.01

    # read fields
    readbytes!(fid, buffer)
    fields = Float64.(reinterpret(Cfloat, buffer))

    # get 2pmarginals
    npairs = binomial(nsites, 2)
    ncouplings = npairs * ncodes * ncodes
    buffer = zeros(UInt8, sizeof(Cfloat)*ncouplings)
    readbytes!(fid, buffer)

    # the sum of the 2pmarginals should be equal to binomial(nsites, 2)
    @assert (sum(reinterpret(Cfloat, buffer)) - npairs) < 0.01

    # read couplings
    readbytes!(fid, buffer)
    couplings_flat = Float64.(reinterpret(Cfloat, buffer))

    # close file
    close(fid)

    # format correctly
    fields = permutedims(reshape(fields, ncodes, nsites))

    couplings = zeros(npairs, ncodes, ncodes)
    k = 1
    for p in 1:npairs
        for a in 1:ncodes
            for b in 1:ncodes
                couplings[p, a, b] = couplings_flat[k]
                k += 1
            end
        end
    end

    if isfile(out_file)
        rm(out_file)
    end
    h5write(out_file, "couplings", couplings)
    h5write(out_file, "fields", fields)
    h5write(out_file, "constant", 0.0)

end

plmc_file = ARGS[1]
out_file = ARGS[2]

if !isfile(plmc_file)
    println("File not found: $plmc_file")
    exit()
end

parse_plmc(plmc_file, out_file)

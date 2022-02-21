using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using DCAUtils
using Statistics

using Distances

function get_seq_len(fasta)
    # ATTENTION: ASSUMES NO LINE BREAKS IN SEQUENCES

    seq_len = 0
    open(fasta) do fid
        for (ind, line) in enumerate(eachline(fid))
            if ind == 1
                @assert line[1]  == '>'
                continue
            end
            if ind == 2
                seq_len = length(strip(line))
                continue
            end
            if ind == 3
                @assert line[1] == '>'
                break
            end
        end
    end

    return seq_len
end

function create_hamming_distances(data_dir)

    dist = Hamming()

    train_fastas = filter(s->endswith(s, "a2m.train"), readdir(data_dir))

    seq_len = readlines

    for train_fasta in train_fastas

        println("Doing $train_fasta")

        test_fasta = replace(train_fasta, "a2m.train" => "a2m.test")

        train_fasta_path = joinpath(data_dir, train_fasta)
        test_fasta_path = joinpath(data_dir, test_fasta)

        if !isfile(train_fasta_path)
            error("cannot find $train_fasta_path")
        end

        if !isfile(test_fasta_path)
            error("cannot find $test_fasta_path")
        end

        # get number of test sequences for consisteny checks later
        ntest = parse(Int, read(`grep -c ">" $test_fasta_path`, String))

        seq_len = get_seq_len(train_fasta_path)

        train_seqs = read_fasta_alignment(train_fasta_path, 1.0)
        test_seqs = read_fasta_alignment(test_fasta_path, 1.0)

        hmd = pairwise(dist, train_seqs, test_seqs, dims=2)

        mean_hmds = vec(mean(hmd, dims=1))
        min_hmds = vec(minimum(hmd, dims=1))

        mean_min_hmds = mean(min_hmds)
        println("mean minimum hamming distance = $mean_min_hmds")

        @assert length(mean_hmds) == ntest

        # assert that there are no zeros
        @assert minimum(min_hmds) > 0

        out_path = replace(train_fasta_path, ".train"=>".train_test_hamming")

        @assert out_path != train_fasta_path

        open(out_path, "w") do fid
            for i in 1:length(mean_hmds)
                s = string.([mean_hmds[i], mean_hmds[i]/seq_len, min_hmds[i], min_hmds[i]/seq_len])
                print(fid, join(s, " "))
                print(fid, '\n')
              end
        end

    end

end

data_dir=ARGS[1]

if !isdir(data_dir)
    error("$data_dir does not exist")
else
    create_hamming_distances(data_dir)
end


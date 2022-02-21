#/bin/bash

set -e
if [[ "$(git rev-parse --show-toplevel)" != "$(pwd)" ]]; then
    echo "Please run from the git root folder of the project"
    exit 1
fi

# this ignores packages not in the current project in order to keep things isolated
export JULIA_LOAD_PATH=@

nsamples=10000000
sample_batch_size=10000
for fasta_path in $(ls data/*train); do
    model_path="models/$(basename $fasta_path).ardca.jld"
    if [ -f ${model_path} ]; then
        echo "Skipping ${fasta_path} since ${model_path} exists"
    else
        echo "Training on ${fasta_path}"
        julia --project=. -t $(nproc --all) ./ardca/ardca.jl train --fasta_path=${fasta_path} --model_path=${model_path}
    fi
    for sample_dist in T U M; do
        sample_file="samples/$(basename $model_path).samples_${sample_dist}.h5"
        if [ -f ${sample_file} ]; then
            echo "Skipping sampling since ${sample_file} exists"
        else
            # sampling from eigen or flat
            if [ "${sample_dist}" != "T" ]; then
                echo "Sampling ${sample_dist} from ${model_path}"
                julia --project=. ./ardca/ardca.jl sample --model_path=${model_path} --nsamples=${nsamples} --sample_batch_size=${sample_batch_size} --out_file=${sample_file} --sample_dist=${sample_dist}
            # if using train samples we only have to evaluate
            elif [ "${sample_dist}" == "T" ]; then
                echo "Sampling ${sample_dist} from ${model_path}"
                julia --project=. ./ardca/ardca.jl evaluate --model_path=${model_path} --fasta_path=${fasta_path} --out_file=${sample_file}
            fi
        fi
    done
done

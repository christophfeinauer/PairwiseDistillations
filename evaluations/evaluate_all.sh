#!/bin/bash

set -e
if [[ "$(git rev-parse --show-toplevel)" != "$(pwd)" ]]; then
    echo "Please run from the git root folder of the project"
    exit 1
fi

. common/bash_utils.sh
. common/parallel_gpu.sh

export JULIA_LOAD_PATH=@

vae_script=vae/evaluate.py
mlp_script=mlp/evaluate_on_fasta.py

# make lists for evaluation
declare -a model_types
declare -a model_paths
declare -a fasta_paths

MAXJOBS=4
num_jobs="\j"

NGPU=$(nvidia-smi -L | wc -l)
GPU=0

for model_file in $(ls models | shuf); do

    model_path=models/${model_file}

    # exclude evmutation binary files
    if [[ "${model_path}" =~ "binary" ]]; then
        continue
    fi

    if [ -d $model_path ]; then
        continue
    elif [[ "${model_path}" =~ "extracted" ]]; then
        model_type="extracted"
    elif [[ "${model_path}" =~ ardca ]]; then
        model_type="ardca"
    elif [[ "${model_path}" =~ vae ]]; then
        model_type="vae"
    elif [[ "${model_path}" =~ mlp ]]; then
        model_type="mlp"
    elif [[ "${model_path}" =~ evmutation ]]; then
        model_type="evmutation"
    else
        echo "could not determine model type for ${model_path}"
        continue
    fi

    # define suffices we want to evaluate on
    suffices=("exp" "test")

    # loop over datasets
    for suffix in "${suffices[@]}"; do
        for fasta_path in $(ls data/*${suffix}); do
            while (( ${num_jobs@P} >= MAXJOBS )); do
                wait -n
            done

            fasta_path_basename=$(basename $fasta_path)
            if ! [[ "$(get_gene_species $model_path)" =~ "$(get_gene_species $fasta_path_basename)" ]]; then
                continue
            fi

            out_file=evaluations/$(basename $model_path).eval_${suffix}
            if [ -s "$out_file" ]; then
                if [ "$out_file" -nt "$model_file" ]; then
                    echo "Not doing $model_file since $out_file is exists and is newer than model file"
                    continue
                fi
            fi
            echo "Evaluating $model_path on ${fasta_path_basename}"
            if [ "${model_type}" == "extracted" ]; then

                cmd="julia --project=. extracted/evaluate_extracted.jl ${model_path} ${fasta_path}"
                $cmd > $out_file

            fi
            if [ "${model_type}" == "evmutation" ]; then

                cmd="julia --project=. extracted/evaluate_extracted.jl ${model_path} ${fasta_path}" 
                $cmd > ${out_file}

            fi
            if [ "${model_type}" == "ardca" ]; then

                cmd="julia --project=. ardca/ardca.jl evaluate --model_path ${model_path} --fasta_path ${fasta_path}"
                $cmd > ${out_file}

            fi
            if [ "${model_type}" == "vae" ]; then

                latent_dim=$(awk -F'.' '{split($6, a, "_"); print a[2]; }' <(echo "$model_file"))
                num_hidden_units=$(awk -F'.' '{split($9, a, "_"); print a[2]; }' <(echo "$model_file"))
                cmd="python $vae_script --latent_dim=${latent_dim} --model_path ${model_path} --batch_size=40 --fasta_path ${fasta_path} --samples=5000 --num_hidden_units=${num_hidden_units}"
                parallel_gpu "$cmd" > ${out_file}

            fi
            if [ "${model_type}" == "mlp" ]; then

                cmd="python $mlp_script --model_path ${model_path} --fasta ${fasta_path}"
                parallel_gpu "$cmd" > ${out_file}

            fi
        done
    done
done

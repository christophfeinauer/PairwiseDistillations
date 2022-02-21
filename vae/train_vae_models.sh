#/bin/bash

. common/parallel_gpu.sh

set -e

train_script=vae/train_on_fasta.py
sample_script=vae/sample.py
evaluate_script=vae/evaluate.py
epochs=10000
nsamples=100000
elbo_samples=5000

SEED=1

weight_decays=(0.001 0.005 0.01 0.05 0.1)
latent_dims=(5 10 20 40 80 120)
num_hidden_units_vec=(40 80 100 120 140 160)

# train models (sampling is done in a second for loop in order to enable naive parallelism - else we try to sample from models that are still training)
for fasta_path in $(ls data/*train); do
    for weight_decay in ${weight_decays[@]}; do
        for latent_dim in ${latent_dims[@]}; do
            for num_hidden_units in ${num_hidden_units_vec[@]}; do

                out_file=models/$(basename $fasta_path).vae.latentdim_${latent_dim}.weightdecay_${weight_decay}.numhiddenunits_${num_hidden_units}.pt

                cmd="python ${train_script} --seed=${SEED} --latent_dim=${latent_dim} --fasta_train_path=${fasta_path} --out_file=${out_file} --epochs ${epochs} --weight_decay ${weight_decay} --num_hidden_units=${num_hidden_units}"
                if [ -s "${out_file}" ]; then
                    echo "Output file ${out_file} exists and is non-empty - skipping"
                else
                    echo $out_file
                    parallel_gpu "$cmd"
                fi

            done
        done
    done
done

# create samples only for latent_dim = 5 and num_hidden_units = 40
latent_dim=5
num_hidden_units=40
for fasta_path in $(ls data/*train); do
    for weight_decay in ${weight_decays[@]}; do
        # create samples only for hidden size = 40 and latent dim = 5
        for sample_dist in M U T; do

            # out file for the model
            out_file=models/$(basename $fasta_path).vae.latentdim_${latent_dim}.weightdecay_${weight_decay}.numhiddenunits_${num_hidden_units}.pt

            # out file for the samples
            out_file_samples=samples/$(basename $out_file).samples_${sample_dist}.h5

            if [ ${sample_dist} == "T" ]; then
                cmd="python ${evaluate_script} --model_path=${out_file} --out_file=${out_file_samples} --fasta_path=${fasta_path} --samples=${elbo_samples} --latent_dim=${latent_dim} --num_hidden_units=${num_hidden_units}"
            else
                cmd="python ${sample_script} --seed=${SEED} --latent_dim=${latent_dim} --model_path=${out_file} --out_file=${out_file_samples} --nsamples ${nsamples} --sample_dist=${sample_dist} --elbo_samples=${elbo_samples} --num_hidden_units=${num_hidden_units}"
            fi

            if [ -s "${out_file_samples}" ]; then
                echo "Output file ${out_file_samples} exists and is non-empty - skipping"
            else
                parallel_gpu "$cmd"
            fi
        done
    done
done

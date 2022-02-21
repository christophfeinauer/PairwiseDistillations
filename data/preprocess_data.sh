#bin/bash

. common/bash_utils.sh

SEED=1 

set -e
if [[ "$(git rev-parse --show-toplevel)" != "$(pwd)" ]]; then
    echo "Please run from the git root folder of the project"
    exit 1
fi

# create train test split
for fasta_path in $(ls data/*a2m); do
    if [[ -s "${fasta_path}".train ]] && [[ -s "${fasta_path}".test ]]; then
       echo "skipping train/test creation for ${fasta_path} since all outputs exist"
       continue
    fi
    echo "creating ${fasta_path}.train and ${fasta_path}.test"
    python data/train_test_split.py "${fasta_path}" --seed=${SEED}
done

# create alignments with experimental data
for fasta_path in $(ls data/*a2m); do

    fasta_file=$(basename $fasta_path)

    # search for matching csv file
    gene_species=$(get_gene_species $fasta_file)
    csv_path_match=""
    for csv_path in $(ls data/*csv); do
        csv_file=$(basename $csv_path)
        if [[ $csv_file == "$gene_species"* ]]; then
            csv_path_match=$csv_path
        fi
    done
    if [ -z "$csv_path_match" ]; then
        echo "cannot find csv file for ${fasta_path}"
        exit 1
    fi
    out_file="${fasta_path}".exp
    if [ -s "${out_file}" ]; then
        echo "skipping exp alignment creation since ${out_file} exists"
        continue
    fi
    echo "creating $out_file"
    python data/create_exp_alignment.py --fasta_path "${fasta_path}" --csv_file="${csv_path_match}" --out_file="${out_file}"

done

# create hamming distance data
julia --project=. ./data/hamming_distances.jl ./data

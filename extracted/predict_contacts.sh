#/bin/bash

pdb_path="./data/1PIN.pdb"
fasta_path="./data/YAP1_HUMAN_1_b0.5.a2m"
chain=A

set -x
set -e


for model_path in $(ls ./models/YAP*extracted* | grep -v IND) './models/YAP1_HUMAN_1_b0.5.a2m.train.ardca.jld'; do
    roc_out_file=./evaluations/$(basename $model_path).pdb_$(basename $pdb_path).roc
    cmap_out_file=./evaluations/$(basename $model_path).pdb_$(basename $pdb_path).cmap
    if [ -f ${roc_out_file} ]; then
        echo "skipping ${model_path} since out file(s) exist"
        continue
    fi
    julia ./extracted/predict_contacts.jl --fasta_path=${fasta_path} --model_path=${model_path} --pdb_path=${pdb_path} --chain ${chain} --roc_out_file=${roc_out_file} --cmap_out_file=${cmap_out_file}
done

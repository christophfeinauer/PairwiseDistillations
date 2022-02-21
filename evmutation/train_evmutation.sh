#/bin/bash

. common/set_fastas.sh

set -x
set -e

train_exe=evmutation/plmc/bin/plmc
convert_script=evmutation/plmc_to_extracted.jl
alphabet='ACDEFGHIKLMNPQRSTVWY-' 

# training parameters
lh=0.01
q=21

for fasta_path in $(ls data/*train); do

    # output file for plmc (need to convert later)
    out_file_plmc=models/$(basename $fasta_path).evmutation.binary

    # final output file
    out_file=models/$(basename $fasta_path).evmutation.h5

    # get sequence length
    seq_len="$(python common/read_fasta.py --print_seq_len --fasta_path ${fasta_path})"

    # calculate the regularization for the coupling (see EVMutation paper)
    le=$(python -c "print(0.01*${q}*(${seq_len}-1))")

    # assemble cmd
    cmd="${train_exe} -o ${out_file_plmc} -lh ${lh} -le ${le} -a "${alphabet}" ${fasta_path}"
    if [ -s "${out_file}" ]; then
        echo "Output file ${out_file} exists and is non-empty - skipping"
    else
        $cmd
    fi
    julia --project=. "${convert_script}" "${out_file_plmc}" "${out_file}"

done

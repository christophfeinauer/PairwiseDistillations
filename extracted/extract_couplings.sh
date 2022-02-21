#/bin/bash

set -e 
set -x

if [[ "$(git rev-parse --show-toplevel)" != "$(pwd)" ]]; then 
    echo "Please run from the git root folder of the project"
    exit 1
fi

# import parallel_gpu
. common/parallel_gpu.sh
NGPU=1


# need to set lang for seq
LANG=en_US

set -e

extract_script=extracted/extract_couplings_from_samples.py

# extraction specifications
batch_size_sgd=10000
batch_size_zs=1000
alpha=0.99

for fasta_path in $(ls data/*train); do

     fasta_file=$(basename $fasta_path)

     for model_path in $(ls -p ./models/${fasta_file}* | grep -v "extracted"); do

         if [ ! -s $model_path ]; then
             echo "${model_path} does not exist, skipping"
         fi

         samples_file_U="samples/$(basename $model_path).samples_U.h5"
         if [ ! -f ${samples_file_U} ]; then
             echo "skipping extraction on ${model_path} since ${samples_file_U} does not exist"
             continue
         fi
         for dist in T M U; do
             for extracted_type in IND PW; do
                out_file=${model_path}.extracted_${extracted_type}_${dist}.h5
                if [ "${dist}" == U ]; then
                    cmd="python $extract_script --samples_file_U=${samples_file_U} --out_file=${out_file} --batch_size_sgd=${batch_size_sgd} --batch_size_zs=${batch_size_zs} --model_type=${extracted_type}"
                else
                    samples_file_dist="samples/$(basename $model_path).samples_${dist}.h5"
                    if [ ! -f ${samples_file_dist} ]; then
                       echo "skipping extraction on ${dist} since ${samples_file_dist} does not exist"
                       continue
                    fi
                    cmd="python $extract_script --alpha=${alpha} --samples_file_dist=${samples_file_dist} --samples_file_U=${samples_file_U} --out_file=${out_file} --batch_size_sgd=${batch_size_sgd} --batch_size_zs=${batch_size_zs} --model_type=${extracted_type}"
                fi
                if [ -s $out_file ]; then
                    echo "Skipping extraction couplings from ${model_path} since ${out_file} exists"
                else
                    echo "Extracting couplings on ${model_path}"
                    parallel_gpu "$cmd"
                fi
            done
        done
    done
done

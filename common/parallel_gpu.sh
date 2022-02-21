num_jobs="\j"
NGPU=$(nvidia-smi -L | wc -l)
MAXJOBS=$(( NGPU ))
GPU=0

parallel_gpu() 
{
    while (( ${num_jobs@P} >= MAXJOBS )); do
        wait -n
    done
    export CUDA_VISIBLE_DEVICES=$GPU
    eval $1 &
    GPU=$((GPU+1))
    GPU=$((GPU%NGPU))
}

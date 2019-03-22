#!/bin/bash
#SBATCH -t 0
#SBATCH --mem-per-cpu=10G
#SBATCH --nodes=1
#SBATCH --mem=30g
#SBATCH --job-name="gs-cpu"
#SBATCH --cpus-per-task 4
# environments
export PATH=/projects/tir1/users/chuntinz/research/syn_emb/tools/miniconda3/bin:$PATH

export CUDA_HOME="/opt/cuda/9.1"
export PATH="/opt/cuda/9.1/bin/:$PATH"
export CUDNN_HOME="/opt/cudnn/cuda-9.1/7.1/cuda/"

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64/:${CUDNN_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDA_HOME}/lib64/:${CUDNN_HOME}/lib64:$LIBRARY_PATH

export LD_LIBRARY_PATH="/opt/gcc/5.4.0/lib64/:$LD_LIBRARY_PATH"

lang=${1}
MODELNAME=${lang}_gs_max_var
cp $0 logs/${MODELNAME}.sh

./multift skipgram -input "/projects/tir1/users/chuntinz/research/syn_emb/data/files/zr_${lang}/plain/data.txt" -output modelfiles/${MODELNAME}  -lr 1e-2 -dim 300 \
    -ws 5 -epoch 6 -minCount 5 -loss gs -bucket 2000000 -clear_prog 0.1 \
    -minn 3 -maxn 6 -thread 5 -t 1e-5 -lrUpdateRate 1000 -var 1 -var_scale 5e-2 -margin 1 -notlog 0 -c 0.001 -min_logvar 0.01 -max_logvar 2.0 2>&1 | tee logs/${MODELNAME}.log

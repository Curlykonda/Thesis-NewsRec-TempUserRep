#!/bin/bash
#SBATCH --job-name=train_npa_keras
#SBATCH -n 6
#SBATCH -t 06:00:00
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=64G


module load pre2019
module load Miniconda3/4.3.27
#module load cuDNN/7.0.5-CUDA-9.0.176
#module load NCCL/2.0.5-CUDA-9.0.176
#module load Python

source activate thesis-user-modelling

workdir=$PWD

#[ -f /etc/resolv.conf ] && echo "File exists" || echo "File does NOT exist"

cd $workdir/train_scripts

python --version

python -u train_npa.py --word_emb_path="../embeddings/cc.nl.300.bin"

#cp -r $outdir $workdir


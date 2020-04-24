#!/bin/bash
#SBATCH --job-name=bert4rec_ML20m
#SBATCH -n 4
#SBATCH -t 06:00:00
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

workdir=$PWD

python --version
#srun -n 2 -t 00:30:00 --pty bash -il

#data=(  "../datasets/dpg/dev_time_split_most_common/"
#        "../datasets/dpg/dev_time_split_random/")

#embeddings="../embeddings/cc.nl.300.bin"

exp_name="bert4rec_ML20m"


echo $exp_name

#for SEED in {42..43}
#do
  #1
  printf '20\ny\n' | python main.py --template train_bert
#done

#!/bin/bash
#SBATCH --job-name=train_npa_large
#SBATCH -n 8
#SBATCH -t 12:00:00
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

workdir=$PWD

#[ -f /etc/resolv.conf ] && echo "File exists" || echo "File does NOT exist"

cd $workdir/train_scripts

python --version

datapath="../datasets/dpg/medium_time_split_interactions/"
embeddings="../embeddings/cc.nl.300.bin"

for SEED in {42..45}
do

python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="med_old" --train_method="wu" \
  --batch_size=100 --random_seed=$SEED
done


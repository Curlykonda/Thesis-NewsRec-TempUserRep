#!/bin/bash
#SBATCH --job-name=run_wu_npa
#SBATCH -n 4
#SBATCH -t 01:00:00
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

#srun -n 2 -t 00:30:00 --pty bash -il

datapath="../datasets/dpg/medium_time_split_interactions/"
embeddings="../embeddings/cc.nl.300.bin"
train="wu"
eval="wu"
interest_extractor="None"

for SEED in {42..46}
do
  #1
  python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="van_npa" --train_method=$train \
    --eval_method=$eval --random_seed=$SEED --interest_extractor=$interest_extractor \
    --max_hist_len=50 --max_news_len=30 --candidate_generation="neg_sampling" --neg_sample_ratio=4 \
    --lambda_l2=0 --n_epochs=20

done
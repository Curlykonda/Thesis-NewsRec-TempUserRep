#!/bin/bash
#SBATCH --job-name=train_npa_dev
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

datapath="../datasets/dpg/dev_time_split/"
embeddings="../embeddings/cc.nl.300.bin"

python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="dev" --train_method="wu"
#python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="dev_cut" --train_method="pos_cut_off" \
#  --test_act_func="softmax"
#python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="dev_cut" --train_method="pos_cut_off" \
#  --interest_extractor="gru" --test_act_func="softmax"
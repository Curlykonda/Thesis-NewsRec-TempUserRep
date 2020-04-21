#!/bin/bash
#SBATCH --job-name=exp_npa_dev
#SBATCH -n 8
#SBATCH -t 08:00:00
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

datapath="../datasets/dpg/dev_time_split/"
embeddings="../embeddings/cc.nl.300.bin"

for SEED in {42..45}
do
  #1
  python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="dev_wu" --train_method="wu" \
    --random_seed=$SEED
  #2
  python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="dev_cut" --train_method="pos_cut_off" \
    --random_seed=$SEED --test_act_func="softmax"
  #3a
  python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="dev_wu_l2-5e3" --train_method="wu" \
    --random_seed=$SEED --lambda_l2=0.005 --test_act_func="softmax"

  #4a
  python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="dev_cut_l2-5e3" --train_method="pos_cut_off" \
    --random_seed=$SEED --lambda_l2=0.005 --test_act_func="softmax"
  #4b
  python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="dev_cut_l2-5e4" --train_method="pos_cut_off" \
    --random_seed=$SEED --lambda_l2=0.0005 --test_act_func="softmax"
  #4c
  python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="dev_cut_l2-2e3" --train_method="pos_cut_off" \
    --random_seed=$SEED --lambda_l2=0.0025 --test_act_func="softmax"

  #5a
  python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="dev_wu_gru" --train_method="wu" \
    --random_seed=$SEED --interest_extractor="gru" --test_act_func="softmax"

  #6a
  python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="dev_cut_gru" --train_method="pos_cut_off" \
    --random_seed=$SEED --interest_extractor="gru" --test_act_func="softmax"

  #7a
  python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="dev_cut_gru_l2-5e3" --train_method="pos_cut_off" \
    --random_seed=$SEED --interest_extractor="gru" --lambda_l2=0.005 --test_act_func="softmax"

  #7b
  python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="dev_cut_gru_l2-5e4" --train_method="pos_cut_off" \
    --random_seed=$SEED --interest_extractor="gru" --lambda_l2=0.0005 --test_act_func="softmax"

  #7c
  python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="dev_cut_gru_l2-2e3" --train_method="pos_cut_off" \
    --random_seed=$SEED --interest_extractor="gru" --lambda_l2=0.0025 --test_act_func="softmax"

done
#python -u train_npa.py --data_path="../datasets/dpg/dev/" --word_emb_path="../embeddings/cc.nl.300.bin" --weight_decay=0.0001 --test_w_one=1 --random_seed=13 --exp_name="dev-wd"
#python -u train_npa.py --data_path="../datasets/dpg/dev/" --word_emb_path="../embeddings/cc.nl.300.bin" --weight_decay=0.0001 --test_w_one=1 --random_seed=102 --exp_name="dev-wd"



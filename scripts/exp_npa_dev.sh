#!/bin/bash
#SBATCH --job-name=exp_npa_dev
#SBATCH -n 4
#SBATCH -t 02:00:00
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

for SEED in {42..45}
do

  python -u train_npa.py --data_path="../datasets/dpg/dev_time_split/" --word_emb_path="../embeddings/cc.nl.300.bin" --exp_name="dev" --random_seed=$SEED

  python -u train_npa.py --data_path="../datasets/dpg/dev_time_split/" --word_emb_path="../embeddings/cc.nl.300.bin" --exp_name="dev-softm" \
                    --eval_method="custom" --train_act_func="softmax" --test_act_func="softmax" --random_seed=$SEED

  python -u train_npa.py --data_path="../datasets/dpg/dev_time_split/" --word_emb_path="../embeddings/cc.nl.300.bin" --exp_name="dev-sig" \
                    --eval_method="custom" --train_act_func="sigmoid" --test_act_func="sigmoid" --random_seed=$SEED

  python -u train_npa.py --data_path="../datasets/dpg/dev_time_split/" --word_emb_path="../embeddings/cc.nl.300.bin" --weight_decay=0.00001 --exp_name="dev-wd1e5" --eval_method="custom" --random_seed=$SEED

  python -u train_npa.py --data_path="../datasets/dpg/dev_time_split/" --word_emb_path="../embeddings/cc.nl.300.bin" --weight_decay=0.0001 --exp_name="dev-wd1e4" --eval_method="custom" --random_seed=$SEED

  python -u train_npa.py --data_path="../datasets/dpg/dev_time_split/" --word_emb_path="../embeddings/cc.nl.300.bin" --weight_decay=0.001 --exp_name="dev-wd1e3" --eval_method="custom" --random_seed=$SEED

done
#python -u train_npa.py --data_path="../datasets/dpg/dev/" --word_emb_path="../embeddings/cc.nl.300.bin" --weight_decay=0.0001 --test_w_one=1 --random_seed=13 --exp_name="dev-wd"
#python -u train_npa.py --data_path="../datasets/dpg/dev/" --word_emb_path="../embeddings/cc.nl.300.bin" --weight_decay=0.0001 --test_w_one=1 --random_seed=102 --exp_name="dev-wd"



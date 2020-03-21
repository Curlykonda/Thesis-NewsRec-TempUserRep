#!/bin/bash
#SBATCH --job-name=train_npa_torch
#SBATCH -n 4
#SBATCH -t 00:30:00
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
python -u train_npa.py --word_emb_path="../embeddings/cc.nl.300.bin"
python -u train_npa.py --word_emb_path="../embeddings/cc.nl.300.bin" --log_method='batches'
python -u train_npa.py --word_emb_path="../embeddings/cc.nl.300.bin" --test_w_one=1
python -u train_npa.py --word_emb_path="../embeddings/cc.nl.300.bin" --log_method='batches' --test_w_one=1
python -u train_npa.py --word_emb_path="../embeddings/cc.nl.300.bin" --eval_method='softmax'
python -u train_npa.py --word_emb_path="../embeddings/cc.nl.300.bin" --eval_method='softmax' --log_method='batches'

python -u train_npa.py --word_emb_path="../embeddings/cc.nl.300.bin" --random_seed=44
python -u train_npa.py --word_emb_path="../embeddings/cc.nl.300.bin" --log_method='batches' --random_seed=44
python -u train_npa.py --word_emb_path="../embeddings/cc.nl.300.bin" --test_w_one=1 --random_seed=44
python -u train_npa.py --word_emb_path="../embeddings/cc.nl.300.bin" --log_method='batches' --test_w_one=1 --random_seed=44
python -u train_npa.py --word_emb_path="../embeddings/cc.nl.300.bin" --eval_method='softmax' --random_seed=44
python -u train_npa.py --word_emb_path="../embeddings/cc.nl.300.bin" --eval_method='softmax' --log_method='batches' --random_seed=44

#python -u train_npa.py --word_emb_path="../embeddings/cc.nl.300.bin" --bce_logits=1


#python -u train_npa.py --word_emb_path="../embeddings/cc.nl.300.bin" --bce_logits=1 --log_method='batches'

#cp -r $outdir $workdir


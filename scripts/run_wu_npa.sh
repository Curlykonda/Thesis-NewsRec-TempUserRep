#!/bin/bash
#SBATCH --job-name=run_wu_npa
#SBATCH -n 4
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

#srun -n 2 -t 00:30:00 --pty bash -il

datapath="../datasets/dpg/medium_time_split_most_common/"
embeddings="../embeddings/cc.nl.300.bin"
train="wu"
eval="wu"

echo $datapath

for SEED in {42..43}
do
  #1
  python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="med_vanilla_npa" \
  --npa_variant="vanilla" --random_seed=$SEED --n_epochs=10 --batch_size=100 --train_method=$train --eval_method=$eval
done

datapath="../datasets/dpg/medium_time_split_random/"
echo $datapath

for SEED in {42..43}
do
  #1
  python -u train_npa.py --data_path=$datapath --word_emb_path=$embeddings --exp_name="med_vanilla_npa" \
  --npa_variant="vanilla" --random_seed=$SEED --n_epochs=10 --batch_size=100 --train_method=$train --eval_method=$eval
done
#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=henning.bartsch@student.uva.nl
source activate master-thesis
for lr in 0.1 0.01 0.001
do
for schedule_gamma in 0.8 0.4 0.2 0.13 0.1
do
for schedule_freq in 15 10 5
do
python classify_mnist.py --model=lenet --lr=${lr} --schedule --schedule_gamma=${schedule_gamma} --schedule_freq=${schedule_freq} --n_epochs=50
done
done
done
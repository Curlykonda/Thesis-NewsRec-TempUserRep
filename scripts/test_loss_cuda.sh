#!/bin/bash
#SBATCH --job-name=test_losses
#SBATCH -t 00:05:00
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

workdir=$PWD

cd $workdir/train_scripts

python --version

#srun -n 2 -t 00:30:00 --pty bash -il
python -u test_losses.py

#cp -r $outdir $workdir


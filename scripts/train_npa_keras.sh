#!/bin/bash
#SBATCH --job-name=train_npa_keras
#SBATCH -n 6
#SBATCH -t 06:00:00
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=64G


module load pre2019
module load Miniconda3/4.3.27
source activate npa-keras

workdir=$PWD/

export PYTHONPATH="$PWD:$PYTHONPATH"

#[ -f /etc/resolv.conf ] && echo "File exists" || echo "File does NOT exist"

cd $workdir/train_scripts

python --version

python -u train_npa_keras.py

#cp -r $outdir $workdir


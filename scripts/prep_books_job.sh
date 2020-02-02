#!/bin/bash
#SBATCH --job-name=preprocessing_books
#SBATCH -n 6
#SBATCH -t 06:00:00
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#module purge

module load pre2019
module load Miniconda3/4.3.27
#module load cuDNN/7.0.5-CUDA-9.0.176
#module load NCCL/2.0.5-CUDA-9.0.176
#module load Python

source activate thesis-user-modelling

#echo "Modules loaded"

#outdir="$TMPDIR"/work-hb/datasets/books-pickle
outdir=$PWD/datasets/out-pickle-full

workdir=$PWD

#indir="$TMPDIR"/work-hb/datasets/

#mkdir -p $indir

mkdir -p $outdir

cp $workdir/datasets/Books_5.json.gz "$TMPDIR"

echo "Done copying."

#chmod 777 $indir

cd $workdir/preprocessing

python --version

python -u prep_amazon_books.py --dataset="$TMPDIR"/Books_5.json.gz --pkl_path=$outdir

#cp -r $outdir $workdir


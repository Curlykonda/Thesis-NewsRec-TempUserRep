#!/bin/bash
#SBATCH --job-name=preprocessing_books
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#module purge

module load pre2019
module load Miniconda3/4.3.27
#module load cuDNN/7.0.5-CUDA-9.0.176
#module load NCCL/2.0.5-CUDA-9.0.176
#module load Python

source activate thesis-user-modelling

echo "Modules loaded"

#outdir="$TMPDIR"/work-hb/datasets/books-pickle
outdir=$PWD/datasets/out-pickle

workdir=$PWD

#indir="$TMPDIR"/work-hb/datasets/

#mkdir -p $indir

mkdir -p $outdir

#echo $workdir

#echo "Output"
#echo $outdir

#echo "Input"
#echo $indir

cp $workdir/datasets/Books_5.json.gz "$TMPDIR"

echo "Done copying."

echo "Unzipping.."

gunzip "$TMPDIR"/Books_5.json.gz

#chmod 777 $indir

cd $workdir/preprocessing

python --version

python -u read_json.py --dataset="$TMPDIR"/Books_5.json --pkl_path=$outdir

echo "Done!"

#cp -r $outdir $workdir


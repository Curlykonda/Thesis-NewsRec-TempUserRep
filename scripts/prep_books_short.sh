#!/bin/bash
#SBATCH --job-name=prep_books_short
#SBATCH -n 6
#SBATCH -t 01:00:00
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#module purge

module load pre2019
module load Miniconda3/4.3.27
#module load cuDNN/7.0.5-CUDA-9.0.176
#module load NCCL/2.0.5-CUDA-9.0.176
#module load Python

source activate thesis-user-modelling

#echo "Modules loaded"

#outdir="$TMPDIR"/work-hb/datasets/books-pickle
outdir=$PWD/datasets/out-pickle/

workdir=$PWD

in_pkl=$PWD/datasets/out-pickle/df.pkl

[ -f /etc/resolv.conf ] && echo "File exists" || echo "File does NOT exist"

#if [ -f "$in_pkl" ]; then
#    echo "$in_pkl exists"
#else 
#    echo "File does not exist"
#    echo "Start copying Dataset"
#    cp $workdir/datasets/Books_5.json.gz "$TMPDIR"
	#echo "Done copying."
#fi


#indir="$TMPDIR"/work-hb/datasets/

#mkdir -p $indir

mkdir -p $outdir

#chmod 777 $indir

cd $workdir/preprocessing

python --version

#python -u prep_amazon_books.py --dataset="$TMPDIR"/Books_5.json.gz --pkl_path=$outdir --n_reviews=2
python -u prep_amazon_books.py --dataset="$TMPDIR"/Books_5.json.gz --pkl_path=$outdir --load_pkl=$in_pkl

#cp -r $outdir $workdir


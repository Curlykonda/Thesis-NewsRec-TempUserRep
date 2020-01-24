#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=henning.bartsch@student.uva.nl

source activate master-thesis

WORK_DIR = '$HOME/thesis-user-modelling'

cp '$WORK_DIR/datasets/Books_5.json.gz' '$TEMP_DIR/datasets/Books_5.json.gz'

cd $WORK_DIR/preprocessing

python prep_amazon_books.py --dataset='$TEMP_DIR/datasets/Books_5.json.gz' --pkl_path='$TEMP_DIR/datasets/books-pickle/'

mkdir -p '$WORK_DIR/datasets/books-pickle/'

cp '$TEMP_DIR/datasets/books-pickle/' '$WORK_DIR/datasets/books-pickle/'

#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH -t 00:05:00
#SBATCH -N 1

#SBATCH --mem=32000M


module purge
module load pre2019
module load Miniconda3/4.3.27

source activate thesis-user-modelling

outdir="$TMPDIR"/work-hb/out
workdir=$PWD

cp -r $PWD/scripts/ "$TMPDIR"/work-hb

echo $outdir

#cp $workdir/datasets/Books_5.json.gz "$TMPDIR"/work-hb/datasets/Books_5.json.gz

cd $PWD/preprocessing

python hello.py

#python test.py --dataset=$HOME/datasets/Books_5.json.gz --pkl_path=$HOME/test-hb/
#srun -n 2 -t 00:10:00 --pty bash -il

#mkdir -p $workdir/out


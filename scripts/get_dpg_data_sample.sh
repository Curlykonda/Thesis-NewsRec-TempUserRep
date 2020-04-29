#!/bin/bash
#SBATCH --job-name=prep_dpg_data
#SBATCH -n 4
#SBATCH -t 01:20:00
#SBATCH --mem=60000M

module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

workdir=$PWD

#[ -f /etc/resolv.conf ] && echo "File exists" || echo "File does NOT exist"

cd $workdir/preprocessing

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

datapath="../datasets/DPG_dec19/"
thresh_date="24-12-2019-23-59-59"

  #1
python -u get_dpg_data_sample.py --data_dir=$datapath --save_path=$datapath --size="medium" \
--time_threshold=$thresh_date --item_sample_method="wu"

#python -u get_dpg_data_sample.py --data_dir=$datapath --save_path=$datapath --size="dev" \
#--time_threshold=$thresh_date --item_sample_method="most_common" --overwrite_existing=0
#  #3
#python -u get_dpg_data_sample.py --data_dir=$datapath --save_path=$datapath --size="medium" \
#--time_threshold=$thresh_date --item_sample_method="random" --overwrite_existing=0
#
#python -u get_dpg_data_sample.py --data_dir=$datapath --save_path=$datapath --size="medium" \
#--time_threshold=$thresh_date --item_sample_method="most_common" --overwrite_existing=0
#
#  #5
#python -u get_dpg_data_sample.py --data_dir=$datapath --save_path=$datapath --size="custom" \
#--time_threshold=$thresh_date --n_users=10000 --n_articles=100000 --item_sample_method="most_common"

#python -u get_dpg_data_sample.py --data_dir=$datapath --save_path=$datapath --size="medium" --time_threshold=$thresh_date
#python -u get_dpg_data_sample.py --data_dir=$datapath --save_path=$datapath --size="dev" --time_threshold=$thresh_date

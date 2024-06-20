#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p general
#SBATCH -q public
#SBATCH --gpus=a100:1
#SBATCH --mem=8G
#SBATCH -t 3-00:00:00
#SBATCH -o log_folder/slurm.%j.out
#SBATCH -e log_folder/slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rpaul12@asu.edu
module load mamba
source activate coldiff

#python -u main.py --method papa --paired_dataset ADNI --extra_str b2a --model_save_path /data/amciilab/rpaul12/mri2pet/saved_models/
#python -u main.py --method papa --paired_dataset ADNI --model_save_path ./saved_models_diceRectified_ce_loss/ --lr 0.001 --n_workers 4
python ixi_train.py --time_steps 50 --save_folder ./results_ixi2_fftshift_2000epochs_50t/ --discrete --sampling_routine x0_step_down --train_steps 2000 --kernel_std 0.1 --fade_routine Constant

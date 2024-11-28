#!/bin/sh
#SBATCH --job-name=gpujob
#SBATCH --gpus=a100-80:1

#SBATCH --job-name=bf_gn
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyi.guo@nus.edu.sg
#SBATCH --partition=gpu-long
#SBATCH --time=15:00:00
#SBATCH --mem-per-gpu=20480

python defenses/bf_gn.py --dataset_name myfriends --img_ids amm --attacks metacloak mist





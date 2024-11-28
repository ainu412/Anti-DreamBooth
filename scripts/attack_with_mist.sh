#!/bin/sh
#SBATCH --job-name=gpujob
#SBATCH --gpus=a100-40:1

#SBATCH --job-name=mist
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyi.guo@nus.edu.sg
#SBATCH --partition=gpu-long
#SBATCH --time=15:00:00

python attacks/mist.py --dataset_name myfriends --img_ids amm


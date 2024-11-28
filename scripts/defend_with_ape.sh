#!/bin/sh
#SBATCH --job-name=gpujob
#SBATCH --gpus=a100-80:1

#SBATCH --job-name=ape
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyi.guo@nus.edu.sg
#SBATCH --partition=gpu-long
#SBATCH --time=15:00:00
#SBATCH --mem-per-gpu=20480

# dataset structure
# -myfriends
# --ziyi
# ---set_A
# ----0.jpg
# ----1.jpg
# ----2.jpg
# ----3.jpg
# ---set_B
# ----4.jpg
# ----5.jpg
# ----6.jpg
# ----7.jpg
# ---set_C
# ----8.jpg
# ----9.jpg
# ----10.jpg
# ----11.jpg
# --ziyi_mist
# ---4.jpg
# ---5.jpg
# ---6.jpg
# ---7.jpg

python defenses/ape.py --dataset_name myfriends --img_ids amm --attacks metacloak mist



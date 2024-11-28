# dataset structure
# -celeb20
# --ariana
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
# --ariana_mist
# ---4.jpg
# ---5.jpg
# ---6.jpg
# ---7.jpg
# --ariana_mist_bf
# ---4.jpg
# ---5.jpg
# ---6.jpg
# ---7.jpg

python evaluations/eval.py --dataset_name myfriends --img_ids chengyu --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape adavoc pdmpure




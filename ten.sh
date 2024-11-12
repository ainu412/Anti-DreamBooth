#!/bin/sh
#SBATCH --job-name=gpujob
#SBATCH --gpus=a100-40:1

#SBATCH --job-name=ariana
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyi.guo@nus.edu.sg
#SBATCH --partition=gpu-long
#SBATCH --time=15:00:00

# get 2 celebrities average result + 2 of us average result
# compare above and analyze
# construct a dataset of 5 colleagues


export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana/set_B"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8

srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada
srun rm -r dreambooth-outputs/ariana

#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="celeb20/ariana_aspl"
#export CLASS_DIR="data/class-person"
#export DREAMBOOTH_OUTPUT_DIR="outputs/ASPL/ariana_DREAMBOOTH"
#
#srun accelerate launch train_dreambooth.py \
#  --pretrained_model_name_or_path=$MODEL_PATH  \
#  --enable_xformers_memory_efficient_attention \
#  --train_text_encoder \
#  --instance_data_dir=$INSTANCE_DIR \
#  --class_data_dir=$CLASS_DIR \
#  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
#  --with_prior_preservation \
#  --prior_loss_weight=1.0 \
#  --instance_prompt="a photo of sks person" \
#  --class_prompt="a photo of person" \
#  --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
#  --resolution=512 \
#  --train_batch_size=2 \
#  --gradient_accumulation_steps=1 \
#  --learning_rate=5e-7 \
#  --lr_scheduler="constant" \
#  --lr_warmup_steps=0 \
#  --num_class_images=100 \
#  --max_train_steps=1000 \
#  --checkpointing_steps=1000 \
#  --center_crop \
#  --mixed_precision=bf16 \
#  --prior_generation_precision=bf16 \
#  --sample_batch_size=8
#srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada
#srun rm -r outputs/ASPL/ariana_DREAMBOOTH

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_aspl_gn"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_aspl_gn/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_aspl_gn

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_aspl_bf"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_aspl_bf/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_aspl_bf

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_aspl_gn_bf"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_aspl_gn_bf/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_aspl_gn_bf

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_aspl_bf_gn"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_aspl_bf_gn/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_aspl_bf_gn

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_aspl_ape"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_aspl_ape/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_aspl_ape

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_aspl_ada"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_aspl_ada/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_aspl_ada

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_aspl_diffpure"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_aspl_diffpure/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_aspl_diffpure



export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_glaze"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_glaze/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_glaze

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_glaze_gn"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_glaze_gn/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_glaze_gn

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_glaze_bf"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_glaze_bf/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_glaze_bf

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_glaze_gn_bf"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_glaze_gn_bf/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_glaze_gn_bf

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_glaze_bf_gn"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_glaze_bf_gn/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_glaze_bf_gn

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_glaze_ape"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_glaze_ape/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8

srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_glaze_ape

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_glaze_ada"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_glaze_ada/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_glaze_ada

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_glaze_diffpure"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_glaze_diffpure/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_glaze_diffpure



export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_metacloak"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_metacloak/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_metacloak

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_metacloak_gn"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_metacloak_gn/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_metacloak_gn

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_metacloak_bf"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_metacloak_bf/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_metacloak_bf
export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_metacloak_gn_bf"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_metacloak_gn_bf/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_metacloak_gn_bf

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_metacloak_bf_gn"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_metacloak_bf_gn/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_metacloak_bf_gn
export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_metacloak_ape"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_metacloak_ape/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_metacloak_ape

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_metacloak_ada"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_metacloak_ada/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_metacloak_ada

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_metacloak_diffpure"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_metacloak_diffpure/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_metacloak_diffpure




export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_mist"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_mist/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_mist

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_mist_gn"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_mist_gn/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_mist_gn

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_mist_bf"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_mist_bf/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_mist_bf
export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_mist_gn_bf"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_mist_gn_bf/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_mist_gn_bf

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_mist_bf_gn"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_mist_bf_gn/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_mist_bf_gn
export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_mist_ape"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_mist_ape/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_mist_ape

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_mist_ada"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_mist_ada/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8
srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_mist_ada

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="celeb20/ariana_mist_diffpure"
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/ariana_mist_diffpure/"

srun accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person; a selfie of sks person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --checkpointing_steps=1000 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8

srun python evaluations/eval.py --dataset_name celeb20 --img_ids ariana --attacks aspl glaze metacloak mist --defenses bf gn bf_gn gn_bf diffpure ape ada

srun rm -r dreambooth-outputs/ariana_mist_diffpure






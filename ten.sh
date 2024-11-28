#!/bin/sh
#SBATCH --job-name=gpujob
#SBATCH --gpus=a100-40:1

#SBATCH --job-name=chengyu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyi.guo@nus.edu.sg
#SBATCH --partition=gpu-long
#SBATCH --time=15:00:00

# get 2 celebrities average result + 2 of us average result
# compare above and analyze
# construct a dataset of 5 colleagues


export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="dataset/myfriends/chengyu/set_B"
export CLASS_DIR="dataset/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu/"

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
  --inference_prompt="a photo of sks person" \
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


export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="dataset/myfriends/chengyu_aspl"
export CLASS_DIR="dataset/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_aspl/"

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
  --inference_prompt="a photo of sks person" \
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



#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_aspl_gn"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_aspl_gn/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_aspl_bf"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_aspl_bf/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_aspl_gn_bf"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_aspl_gn_bf/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_aspl_bf_gn"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_aspl_bf_gn/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_aspl_ape"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_aspl_ape/"
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
#  --inference_prompt="a photo of sks person" \
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


export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="dataset/myfriends/chengyu_aspl_adavoc"
export CLASS_DIR="dataset/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_aspl_adavoc/"

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
  --inference_prompt="a photo of sks person" \
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

#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_aspl_diffpure"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_aspl_diffpure/"
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
#  --inference_prompt="a photo of sks person" \
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

export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="dataset/myfriends/chengyu_aspl_pdmpure"
export CLASS_DIR="dataset/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_aspl_pdmpure/"

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
  --inference_prompt="a photo of sks person" \
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


#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_glaze"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_glaze/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_glaze_gn"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_glaze_gn/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_glaze_bf"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_glaze_bf/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_glaze_gn_bf"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_glaze_gn_bf/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_glaze_bf_gn"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_glaze_bf_gn/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_glaze_ape"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_glaze_ape/"
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
#  --inference_prompt="a photo of sks person" \
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



export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="dataset/myfriends/chengyu_glaze_adavoc"
export CLASS_DIR="dataset/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_glaze_adavoc/"

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
  --inference_prompt="a photo of sks person" \
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


#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_glaze_diffpure"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_glaze_diffpure/"
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
#  --inference_prompt="a photo of sks person" \
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


export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="dataset/myfriends/chengyu_glaze_pdmpure"
export CLASS_DIR="dataset/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_glaze_pdmpure/"

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
  --inference_prompt="a photo of sks person" \
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

#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_metacloak"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_metacloak/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_metacloak_gn"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_metacloak_gn/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_metacloak_bf"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_metacloak_bf/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_metacloak_gn_bf"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_metacloak_gn_bf/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_metacloak_bf_gn"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_metacloak_bf_gn/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_metacloak_ape"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_metacloak_ape/"
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
#  --inference_prompt="a photo of sks person" \
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


export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="dataset/myfriends/chengyu_metacloak_adavoc"
export CLASS_DIR="dataset/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_metacloak_adavoc/"

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
  --inference_prompt="a photo of sks person" \
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



#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_metacloak_diffpure"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_metacloak_diffpure/"
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
#  --inference_prompt="a photo of sks person" \
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


export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="dataset/myfriends/chengyu_metacloak_pdmpure"
export CLASS_DIR="dataset/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_metacloak_pdmpure/"

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
  --inference_prompt="a photo of sks person" \
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



#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_mist"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_mist/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_mist_gn"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_mist_gn/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_mist_bf"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_mist_bf/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_mist_gn_bf"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_mist_gn_bf/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_mist_bf_gn"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_mist_bf_gn/"
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
#  --inference_prompt="a photo of sks person" \
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
#
#
#
#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_mist_ape"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_mist_ape/"
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
#  --inference_prompt="a photo of sks person" \
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


export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="dataset/myfriends/chengyu_mist_adavoc"
export CLASS_DIR="dataset/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_mist_adavoc/"

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
  --inference_prompt="a photo of sks person" \
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


#export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
#export INSTANCE_DIR="dataset/myfriends/chengyu_mist_diffpure"
#export CLASS_DIR="dataset/class-person"
#export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_mist_diffpure/"
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
#  --inference_prompt="a photo of sks person" \
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


export MODEL_PATH="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="dataset/myfriends/chengyu_mist_pdmpure"
export CLASS_DIR="dataset/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/chengyu_mist_pdmpure/"

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
  --inference_prompt="a photo of sks person" \
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





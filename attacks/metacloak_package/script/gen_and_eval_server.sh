#!/bin/sh
#SBATCH --gpus=a100-40:1


#SBATCH --job-name=beyonce_metacloak
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyi.guo@nus.edu.sg
#SBATCH --mem=25G
#SBATCH --partition=gpu-long

#source activate metacloak

#### Variables that you should set before running the script
export PYTHONPATH="${PYTHONPATH}:/mnt/c/Users/ziyi.guo/Documents/GitHub/MetaCloak"
export ADB_PROJECT_ROOT="/mnt/c/Users/ziyi.guo/Documents/GitHub/MetaCloak"
# setting the GPU id
export CUDA_VISIBLE_DEVICES=0
# for logging , please set your own wandb entity name
export wandb_entity_name=momo
# for reproducibility
export seed=0
# for other models, change the name and path correspondingly
# SD21, SD21base, SD15, SD14
export gen_model_name=SD21base # the model to generate the noise 
export MODEL_ROOT=$ADB_PROJECT_ROOT
export gen_model_path=stabilityai/stable-diffusion-2-1-base
export eval_model_name=SD21base # the model that performs the evaluation on noise 
export eval_model_path=stabilityai/stable-diffusion-2-1-base
export r=11 # the noise level of perturbation
export gen_prompt=sks # the prompt used to craft the noise
export eval_prompt=sks # the prompt used to do dreambooth training
export dataset_name=beyonce # for the Cele, change it to CelebA-HQ-clean
# support method name: metacloak, clean
export method_select=metacloak
# the training mode, can be std or gau
export train_mode=gau 
# if gau is selected, the gauK is the kernel size of guassian filter 
export gauK=7
export eval_gen_img_num=16 # the number of images to generate per prompt 
export round=final # which round of noise to use for evaluation
# the instance name of image, for loop all of them to measure the performance of MetaCloak on whole datasets
export instance_name=0
# this is for indexing and retriving the noise for evaluation
export prefix_name_gen=release
export prefix_name_train=gau
##############################################################


# for later usage 
export model_name=$gen_model_name
# go to main directory
pwdd=$(cd "$(dirname "$0")";pwd)
cd $pwdd
pwd
. methods_config.sh
export step_size=$(echo "scale=2; $r/10" | bc); 
export gen_exp_name_prefix=$prefix_name_gen
export prefix_name_train=$prefix_name_train
export method_hyper_name=$method_name-$method_hyper
export gen_exp_name=$gen_exp_name_prefix-$method_hyper_name
export gen_exp_hyper=dataset-$dataset_name-r-$r-model-$gen_model_name-gen_prompt-$gen_prompt
export OUTPUT_DIR="$ADB_PROJECT_ROOT/exp_data/gen_output/${gen_exp_name}/$gen_exp_hyper/${instance_name}"
export INSTANCE_DIR=$OUTPUT_DIR/noise-ckpt/${round}
export CLEAN_INSTANCE_DIR=$OUTPUT_DIR/image_before_addding_noise/
export INSTANCE_DIR_CHECK=$INSTANCE_DIR


# generate the noise 
bash ./sub/gen/$gen_file_name

## evaluate the noise with dreambooth training
#bash ./sub/eval/generic.sh

echo $(date +%R)


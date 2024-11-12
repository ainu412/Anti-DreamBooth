from deepfloyd_if.pipelines import style_transfer
from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from diffusers.utils import pt_to_pil, load_image
import torch
import argparse
import os
import glob

from huggingface_hub import login

# login()
import torch
torch.cuda.empty_cache()

celeb20_img_id_list = ['ariana', 'beyonce', 'bruce', 'cristiano', 'ellen', 'emma', 'george', 'jackie', 'james',
                       'johnny', 'justin', 'kate', 'leonardo', 'lucy', 'morgan', 'oprah', 'rihanna', 'shah', 'shirley',
                       'taylor']

def run(img_path):
    device = 0
    img_name = os.path.basename(img_path)[:-4] # peacock
    output_dir = os.path.dirname(img_path)
    resampling = "10,0,0,0,0,0,0,0,0,0"
    resampling = "10,10,10,10,10,0,0,0,0,0"
    
    # LOAD IMAGES
    raw_pil_image = load_image(img_path)
    OUT_SHAPE = raw_pil_image.size
    raw_pil_image_mid = raw_pil_image.resize((256, 256))

    # LOAD DEEPFLOYD MODELS
    if_II = IFStageII('IF-II-L-v1.0', device=device)
    if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device)
    t5 = T5Embedder(device=device)
    
    # RUN PURIFICATION
    print(f'Begin to purify {img_path}' + '-' * 10)
    support_noise_less_qsample_steps = 5
    with torch.no_grad():
        result = style_transfer(
            t5=t5, if_I=None, if_II=if_II, if_III=if_III,
            support_pil_img=raw_pil_image,
            style_prompt=[
                'a photo'
            ],
            seed=0,
            if_II_kwargs={ 
                "guidance_scale": 7,
                "sample_timestep_respacing":  resampling,
                "support_noise_less_qsample_steps": support_noise_less_qsample_steps,  # 5, 10, 15
                "low_res": raw_pil_image_mid
            },
            if_III_kwargs={ 
                "guidance_scale": 4.0,
                "sample_timestep_respacing": "50",
            },
            disable_watermark=True
        )
        
        # raw_pil_image.save(args.save_path + 'original.png')
        os.makedirs(f'{output_dir}_pdmpure', exist_ok=True)
        result['III'][0].resize(OUT_SHAPE).save(f'{output_dir}_pdmpure/{img_name}_pdmpure.png')



def cleaning(img_path):
    # if path is a folder containing images
    if os.path.isdir(img_path):
        img_paths = glob.glob(img_path + '/*')
        for p in img_paths:
            run(img_path=p)
    # if path is a single image
    else:
        run(img_path=img_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='myfriends')
    parser.add_argument("--img_ids", type=str, nargs='+', default=['amy', 'kiat', 'qian', 'yuexin', 'ziyi'])
    parser.add_argument("--attacks", type=str, nargs='+', default=['metacloak', 'aspl', 'glaze', 'mist'])
    args = parser.parse_args()

    for attack in args.attacks:
        for img_id in args.img_ids:
            img_path = f"{args.dataset_name}/{img_id}_{attack}"
            cleaning(img_path)



if __name__ == '__main__':
    main()
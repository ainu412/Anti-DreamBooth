import argparse
import glob

import pandas as pd
import os
from PIL import Image

from ism_fdfr import matching_score_genimage_id
from ser_fiq import fiq, fiq_single
from brisques import brisque

prompt = "a_photo_of_sks_person"
# prompts = ["a_dslr_portrait_of_sks_person", "a_photo_of_sks_person", "a_selfie_of_sks_person"]
celeb20_img_id_list = ['ariana', 'beyonce', 'bruce', 'cristiano', 'ellen', 'emma', 'george', 'jackie', 'james',
                       'johnny', 'justin', 'kate', 'leonardo', 'lucy', 'morgan', 'oprah', 'rihanna', 'shah', 'shirley',
                       'taylor']
myfriends_img_id_list = ['amy', 'kiat', 'qian', 'yuexin', 'ziyi']

attacks = ['aspl', 'metacloak', 'mist', 'glaze']
defenses = ['adavoc', 'ape', 'bf', 'bf_gn', 'diffpure', 'gn', 'gn_bf', 'pdmpure']

def check_exist(img, prompt, csv_path):
    if not os.path.exists(csv_path):
        return False

    df = pd.read_csv(csv_path)
    df_cur = df[(df['img'] == img)]
    return not df_cur.empty

def main(args):
    csv_path = 'result/result.csv' if args.generated else 'result/finetune_set.csv'
    num_image_per_folder = 16 if args.generated else 4

    for img_id in args.img_ids:
        for adv_algorithm in args.attacks:
            for img_suffix in ['', f'_{adv_algorithm}'] + [f'_{adv_algorithm}_{cleaning_method}' for cleaning_method in args.defenses]:
                if args.generated:
                    img_path = f"dreambooth-outputs/{img_id}{img_suffix}/checkpoint-1000/dreambooth/{prompt}/"
                else:
                    img_path = f"{args.dataset_name}/{img_id}{img_suffix}" if img_suffix else f"{args.dataset_name}/{img_id}{img_suffix}/set_B"

                # check image path exist or not
                if not os.path.exists(img_path):
                    print(f'img_path {img_path} do not exist')
                    continue

                # check if the folder contains desired amount of images
                if len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))]) < num_image_per_folder:
                    print(f'img_path {img_path} do not have all {num_image_per_folder} images')
                    continue

                # check if this value has been recorded in the form
                if check_exist(img_id + img_suffix, prompt, csv_path):
                    continue

                print(f'processing {img_id}{img_suffix} {prompt}...')
                result_dic = dict()
                result_dic['img'] = img_id + img_suffix

                clean_img_paths = [f"dataset/{args.dataset_name}/{img_id}/set_A",
                                   f"dataset/{args.dataset_name}/{img_id}/set_B",
                                   f"dataset/{args.dataset_name}/{img_id}/set_C"]

                ism, fdfr = matching_score_genimage_id(img_path, clean_img_paths)
                result_dic['ism'] = '%.4f' % ism if ism is not None else 0
                result_dic['fdr'] = '%.4f' % (1 - fdfr)

                df = pd.DataFrame(result_dic, index=[0])
                # append to csv
                if not os.path.exists(csv_path):
                    df.to_csv(csv_path, index=False)
                else:
                    df.to_csv(csv_path, index=False, mode='a', header=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='myfriends')
    parser.add_argument("--img_ids", type=str, nargs='+', default=myfriends_img_id_list)
    parser.add_argument("--attacks", type=str, nargs='+', default=attacks)
    parser.add_argument("--defenses", type=str, nargs='+', default=defenses)
    parser.add_argument("--generated", type=bool, default=True, help='use generated images or finetune input images')

    args = parser.parse_args()
    main(args)

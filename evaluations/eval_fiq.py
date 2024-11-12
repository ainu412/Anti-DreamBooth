import argparse
import time
import glob

import pandas as pd
import os
from PIL import Image

from ism_fdfr import matching_score_genimage_id, matching_score_genimage_id_li
from ser_fiq import fiq, fiq_single
from brisques import brisque

prompts = ["a_photo_of_sks_person"]
# prompts = ["a_dslr_portrait_of_sks_person", "a_photo_of_sks_person", "a_selfie_of_sks_person"]
celeb20_img_id_list = ['ariana', 'beyonce', 'bruce', 'cristiano', 'ellen', 'emma', 'george', 'jackie', 'james',
                       'johnny', 'justin', 'kate', 'leonardo', 'lucy', 'morgan', 'oprah', 'rihanna', 'shah', 'shirley',
                       'taylor']
checkpoints = [20, 40, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]


def check_exist(img, prompt, csv_path):
    if not os.path.exists(csv_path):
        return False

    df = pd.read_csv(csv_path)
    df_cur = df[(df['img'] == img) & (df['prompt'] == prompt)]
    return not df_cur.empty


def check_exist_steps(img, prompt, steps, csv_path):
    if not os.path.exists(csv_path):
        return False

    df = pd.read_csv(csv_path)
    df_cur = df[(df['img'] == img) & (df['prompt'] == prompt) & (df['steps'] == steps)]
    return not df_cur.empty


def check_exist_input(img, csv_path):
    if not os.path.exists(csv_path):
        return False

    df = pd.read_csv(csv_path)
    df_cur = df[(df['img'] == img)]
    return not df_cur.empty


def main(args):
    for img_id in args.img_ids:
        for adv_algorithm in args.attacks:
            for prompt in prompts:
                for img_suffix in ['', f'_{adv_algorithm}'] + [f'_{adv_algorithm}_{cleaning_method}' for cleaning_method
                                                               in args.defenses]:
                    steps = 300 if img_id in ["man", "woman"] else 1000
                    img_path = f"outputs/{adv_algorithm.upper()}/{img_id}_DREAMBOOTH/checkpoint-{steps}/dreambooth/{prompt}" if img_suffix == f'_{adv_algorithm}' and adv_algorithm in [
                        'aspl', 'fsmg', 't-aspl', 't-fsmg', 'e-aspl',
                        'e-fsmg'] else f"dreambooth-outputs/{img_id}{img_suffix}/checkpoint-1000/dreambooth/{prompt}/"

                    if not os.path.exists(img_path):
                        print('img_path', img_path, 'do not exist')
                        continue

                    if len([name for name in os.listdir(img_path) if
                            os.path.isfile(os.path.join(img_path, name))]) < 16:
                        print('img_path', img_path, 'do not have all 16 images')
                        continue

                    csv_path = f'result/result_new_{img_id}.csv'

                    if check_exist(img_id + img_suffix, prompt, csv_path):
                        continue

                    print(f'processing {img_id}{img_suffix} {prompt}...')
                    result_dic = dict()
                    result_dic['img'] = img_id + img_suffix
                    result_dic['prompt'] = prompt

                    clean_img_paths = [f"{args.dataset_name}/{img_id}"] if img_id in ["man", "woman"] else [
                        f"{args.dataset_name}/{img_id}/set_A", f"{args.dataset_name}/{img_id}/set_B",
                        f"{args.dataset_name}/{img_id}/set_C"]

                    ism, fdfr = matching_score_genimage_id(img_path, clean_img_paths)
                    result_dic['ism↑'] = '%.4f' % ism if ism is not None else 0
                    result_dic['fdr↑'] = '%.4f' % (1 - fdfr)
                    # result_dic['fiq↑'] = '%.4f' % fiq(img_path)
                    result_dic['brisque↓'] = '%.4f' % brisque(img_path)

                    df = pd.DataFrame(result_dic, index=[0])
                    # append to csv
                    if not os.path.exists(csv_path):
                        df.to_csv(csv_path, index=False)
                    else:
                        df.to_csv(csv_path, index=False, mode='a', header=False)


def evaluate_steps(csv_path='result/result_steps.csv'):
    for img_id in ["man"]:
        for checkpoint in checkpoints:
            for prompt in ["a_photo_of_sks_person"]:
                img_path = f"dreambooth-outputs/{img_id}/checkpoint-{checkpoint}/dreambooth/{prompt}/"
                if not os.path.exists(img_path):
                    print('img_path', img_path, 'do not exist')
                    continue

                if len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))]) < 4:
                    print('img_path', img_path, 'do not have all 4 images')
                    continue

                if check_exist_steps(img_id, prompt, checkpoint, csv_path):
                    continue

                print(f'processing {img_id}{checkpoint} {prompt}...')
                result_dic = dict()
                result_dic['img'] = img_id
                result_dic['steps'] = checkpoint
                result_dic['prompt'] = prompt

                clean_img_paths = [f"data/{img_id}"]

                ism, fdfr = matching_score_genimage_id(img_path, clean_img_paths)
                result_dic['ism'] = '%.4f' % ism if ism is not None else 0
                result_dic['fdr'] = '%.4f' % (1 - fdfr)
                # result_dic['fiq'] = '%.4f' % fiq(img_path)
                result_dic['brisque'] = '%.4f' % brisque(img_path)

                df = pd.DataFrame(result_dic, index=[0])
                # append to csv
                if not os.path.exists(csv_path):
                    df.to_csv(csv_path, index=False)
                else:
                    df.to_csv(csv_path, index=False, mode='a', header=False)


def evaluate_dreambooth_finetune_input(args):
    for img_id in args.img_ids:
        for adv_algorithm in args.attacks:
            for img_suffix in ['', f'_{adv_algorithm}'] + [f'_{adv_algorithm}_{cleaning_method}' for cleaning_method in
                                                           args.defenses]:
                img_path = f"{args.dataset_name}/{img_id}{img_suffix}" if img_suffix else f"{args.dataset_name}/{img_id}{img_suffix}/set_B"
                if not os.path.exists(img_path):
                    print('img_path', img_path, 'do not exist')
                    continue

                if len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))]) < 4:
                    print('img_path', img_path, 'do not have all 4 images')
                    continue

                if check_exist_input(img_id + img_suffix, args.output_path):
                    continue

                print(f'processing {img_id}{img_suffix}...')
                result_dic = dict()
                result_dic['img'] = img_id + img_suffix

                clean_img_paths = [f"{args.dataset_name}/{img_id}/set_A", f"{args.dataset_name}/{img_id}/set_B", f"{args.dataset_name}/{img_id}/set_C"]

                ism, fdfr = matching_score_genimage_id(img_path, clean_img_paths)
                result_dic['ism'] = '%.4f' % ism if ism is not None else ism
                result_dic['fdr'] = '%.4f' % (1 - fdfr)

                df = pd.DataFrame(result_dic, index=[0])
                # append to csv
                if not os.path.exists(args.output_path):
                    df.to_csv(args.output_path, index=False)
                else:
                    df.to_csv(args.output_path, index=False, mode='a', header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='celeb20')
    parser.add_argument("--img_ids", type=str, nargs='+', default=celeb20_img_id_list)
    parser.add_argument("--attacks", type=str, nargs='+', default=['aspl', 'glaze', 'metacloak', 'mist'])
    parser.add_argument("--defenses", type=str, nargs='+',
                        default=['adavoc', 'ape', 'bf', 'bf_gn', 'diffpure', 'gn', 'gn_bf'])
    parser.add_argument("--output_path", type=str, default="result/result_finetune_input.csv")

    args = parser.parse_args()
    # ism_fdr_images(args)
    evaluate_dreambooth_finetune_input(args)

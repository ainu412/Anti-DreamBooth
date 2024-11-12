import cv2
import argparse
import os
from FaceImageQuality.face_image_quality import SER_FIQ


def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--prompt_path', default='/vinai/quandm7/evaluation_dreambooth/Celeb/5/',
                        help='path to input image file')
    args = parser.parse_args()
    return args

def fiq(prompt_path):
    ser_fiq = SER_FIQ(gpu=0)
    prompt_score = 0
    count = 0

    for img_name in os.listdir(prompt_path):
        if "png" in img_name or "jpg" in img_name:
            img_path = os.path.join(prompt_path, img_name)
            img = cv2.imread(img_path)
            aligned_img = ser_fiq.apply_mtcnn(img)
            if aligned_img is not None:
                score = ser_fiq.get_score(aligned_img, T=100)
                prompt_score += score
                count += 1

    return prompt_score / count

def fiq_single(img_path):
    ser_fiq = SER_FIQ(gpu=0)
    img = cv2.imread(img_path)
    aligned_img = ser_fiq.apply_mtcnn(img)
    if aligned_img is not None:
        score = ser_fiq.get_score(aligned_img, T=100)
        return score

if __name__ == '__main__':
    args = parse_args()
    fiq_score = fiq(args.prompt_path)
    print("FIQ score: {}".format(fiq_score))
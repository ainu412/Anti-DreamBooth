import os
import glob

import numpy as np
from PIL import Image
import torch
from ape_package.models import MnistCNN, CifarCNN, Generator
import argparse
import torchvision.transforms as transforms
from torchvision.utils import save_image


def load_cnn(args):
    if args.data == "mnist":
        return MnistCNN
    elif args.data == "cifar":
        return CifarCNN

PIL2tensor = transform = transforms.Compose([
    transforms.ToTensor()
])

def run(img_path, args, save_imgs='true'):
    img_name = os.path.basename(img_path)[:-4]  # peacock
    output_dir = os.path.dirname(img_path)

    # process on every img
    img_pil = Image.open(img_path)
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    image_tensor = PIL2tensor(img_pil).cuda()

    # load ape-gan
    model_point = torch.load("./CNN/cifar_cnn.tar")
    gan_point = torch.load(args.gan_path)

    CNN = load_cnn(args)

    model = CNN().cuda()
    model.load_state_dict(model_point["state_dict"])

    in_ch = 1 if args.data == "mnist" else 3

    G = Generator(in_ch).cuda()
    G.load_state_dict(gan_point["generator"])

    # clean image
    x_ape = G(image_tensor[np.newaxis,:,:,:])

    # save ape image to folder
    print('shape!', x_ape.shape)
    if save_imgs == 'true':
        os.makedirs(f'{output_dir}_ape', exist_ok=True)
        save_image(x_ape[0], f'{output_dir}_ape/{img_name}_ape.png')


# I don't train the model again based on our face data, I just use it's trained Cifar10CNN model


def main(args, img_path):

    # if path is a folder containing images
    if os.path.isdir(img_path):
        img_paths = glob.glob(img_path + '/*')
        for p in img_paths:
            run(img_path=p, args=args)
    # if path is a single image
    else:
        run(img_path=img_path, args=args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='myfriends')
    parser.add_argument("--img_ids", type=str, nargs='+', default=["ziyi", 'qian', 'jiyan', 'weitsang'])
    parser.add_argument("--attacks", type=str, nargs='+', default=['metacloak', 'aspl', 'glaze', 'mist'])
    parser.add_argument("--data", type=str, nargs='+', default="cifar")
    parser.add_argument("--eps", type=float, default=0.15)
    parser.add_argument("--gan_path", type=str, default="./checkpoint/cifar.tar")
    args = parser.parse_args()

    for img_id in args.img_ids:
        for adv_algorithm in args.attacks:
            img_path = f'{args.dataset_name}/{img_id}_{adv_algorithm}'
            main(args, img_path)
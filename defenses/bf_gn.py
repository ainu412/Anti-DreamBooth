import argparse
import os
import torchvision.transforms as transforms
from typing import Union
import cv2
import torch
from PIL import Image
import numpy as np
import glob


PIL2tensor = transform = transforms.Compose([
    transforms.ToTensor()
])

celeb20_img_id_list = ['ariana', 'beyonce', 'bruce', 'cristiano', 'ellen', 'emma', 'george', 'jackie', 'james',
                       'johnny', 'justin', 'kate', 'leonardo', 'lucy', 'morgan', 'oprah', 'rihanna', 'shah', 'shirley',
                       'taylor']

def _apply_edge_preserve_filter_for_np(_img_np, d=3, sigmaSpace=20, sigmaColor=20):
    assert d % 2 == 1 and d > 0, "d should be odd and positive"
    assert sigmaSpace > 0 and sigmaColor > 0, "sigmaSpace and sigmaColor should be positive"

    _img_np = _img_np / 255.0 if _img_np.dtype == np.uint8 else _img_np
    assert _img_np.dtype == np.float32 or _img_np.dtype == np.float64, f"dtype should be float32, but got {_img_np.dtype}"
    assert _img_np.max() <= 1.0 and _img_np.min() >= 0.0
    _img_np = cv2.bilateralFilter(_img_np.astype(np.float32), d=d, sigmaSpace=sigmaSpace, sigmaColor=sigmaColor)
    return _img_np


def apply_edge_preserve_filter(img: Union[torch.Tensor, np.ndarray, Image.Image],
                               d=3,
                               sigmaSpace=20,
                               sigmaColor=20,
                               ) -> Union[torch.Tensor, np.ndarray, Image.Image]:
    if isinstance(img, torch.Tensor):
        # convert to numpy
        img = img.cpu().numpy()
        img_np = _apply_edge_preserve_filter_for_np(img, d, sigmaSpace, sigmaColor)
        return torch.from_numpy(img_np)
    elif isinstance(img, Image.Image):
        img = np.array(img)
        img_np = _apply_edge_preserve_filter_for_np(img, d, sigmaSpace, sigmaColor)
        img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)
    elif isinstance(img, np.ndarray):
        img_np = _apply_edge_preserve_filter_for_np(img, d, sigmaSpace, sigmaColor)
        return img_np
    else:
        raise ValueError("img type not supported, only support torch.Tensor, np.ndarray, Image.Image")


def tensor2numpy(img_tensor):
    return img_tensor.numpy()


def pil2numpy(img_pil):
    return np.array(img_pil)

def tensor2image(img_tensor):
    img_pil = transforms.ToPILImage()(img_tensor)
    return img_pil


def edge_sanitize_img(adv_img_tensor):
    adv_img_pil = transforms.ToPILImage()(adv_img_tensor)
    sanitized_img = apply_edge_preserve_filter(adv_img_pil, d=3)
    return transforms.ToTensor()(sanitized_img)


def tensor2numpy2tensor(tensor):
    tensor_np = tensor.numpy() * 255
    tensor_np = tensor_np.astype(np.uint8)
    tensor_np = tensor_np / 255
    tensor = torch.from_numpy(tensor_np).float()
    return tensor


def pil2tensor(img_pil):
    img_tensor = transforms.ToTensor()(img_pil)
    return img_tensor


def gn_sanitize_img(adv_img_tensor):
    return torch.clamp(adv_img_tensor + torch.randn_like(adv_img_tensor) * 0.06 - 0.005, 0, 1)





def run(img_path: str = 'sample/peacock.jpg',
        save_imgs: str = 'true',
        ):

    print('cleaning', img_path)
    img_name = os.path.basename(img_path)[:-4] # peacock
    output_dir = os.path.dirname(img_path)

    # process on every img
    img_pil = Image.open(img_path)
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    image_tensor = PIL2tensor(img_pil)

    # use edge preserve filter
    edge_sanitized_img = edge_sanitize_img(image_tensor)
    edge_sanitized_img_pil = tensor2image(edge_sanitized_img)
    if save_imgs == 'true':
        # edge_sanitized_img_pil.save(f'{img_path[:-4]}_bf.png')
        os.makedirs(f'{output_dir}_bf', exist_ok=True)
        edge_sanitized_img_pil.save(f'{output_dir}_bf/{img_name}_bf.png')

    # use gaussian noise
    gn_sanitized_img = gn_sanitize_img(image_tensor)
    gn_sanitized_img_pil = tensor2image(gn_sanitized_img)
    if save_imgs == 'true':
        # gn_sanitized_img_pil.save(f'{img_path[:-4]}_gn.png')
        os.makedirs(f'{output_dir}_gn', exist_ok=True)
        gn_sanitized_img_pil.save(f'{output_dir}_gn/{img_name}_gn.png')

    # use edge preserve filter + gaussian noise
    edge_gn_sanitized_img = gn_sanitize_img(edge_sanitized_img)
    edge_gn_sanitized_img_pil = tensor2image(edge_gn_sanitized_img)
    if save_imgs == 'true':
        # edge_gn_sanitized_img_pil.save(f'{img_path[:-4]}_bf_gn.png')
        os.makedirs(f'{output_dir}_bf_gn', exist_ok=True)
        edge_gn_sanitized_img_pil.save(f'{output_dir}_bf_gn/{img_name}_bf_gn.png')

    # use gaussian noise + edge preserve filter
    gn_edge_sanitized_img = edge_sanitize_img(gn_sanitized_img)
    gn_edge_sanitized_img_pil = tensor2image(gn_edge_sanitized_img)
    if save_imgs == 'true':
        # gn_edge_sanitized_img_pil.save(f'{img_path[:-4]}_gn_bf.png')
        os.makedirs(f'{output_dir}_gn_bf', exist_ok=True)
        gn_edge_sanitized_img_pil.save(f'{output_dir}_gn_bf/{img_name}_gn_bf.png')


def cleaning(img_path):
    # if path is a folder containing images
    if os.path.isdir(img_path):
        img_paths = glob.glob(img_path + '/*')
        if os.path.isfile(img_paths[0]):
            for p in img_paths:
                run(img_path=p)
        # if path is a folder, and contain multiple folders containing images
        elif os.path.isdir(img_paths[0]):
            for img_folder in img_paths:
                img = glob.glob(img_folder + '/*')
                if 'set' in img[0]:
                    continue
                for p in img:
                    run(img_path=p)

    # if path is a single image
    else:
        run(img_path=img_path)


def frame_to_tensor(frame):
    # Convert BGR (OpenCV default format) to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame (a NumPy array) to a PyTorch tensor
    transform = transforms.ToTensor()  # Converts image to [0,1] and orders it as [C, H, W]
    tensor = transform(frame_rgb)

    return tensor


def tensor_to_frame(tensor):
    # Ensure the tensor is on the CPU and detach if necessary (e.g., if coming from GPU)
    if tensor.requires_grad:
        tensor = tensor.detach()
    tensor = tensor.cpu().numpy()  # Convert the tensor to a NumPy array

    # Permute the tensor from (C, H, W) to (H, W, C)
    frame = np.transpose(tensor, (1, 2, 0))

    # Convert from [0, 1] range to [0, 255] and make sure it's an integer type
    frame = (frame * 255).astype(np.uint8)

    # Optional: Convert from RGB to BGR for OpenCV (if needed)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame_bgr

def gn_bf(frame):
    # Example of frame processing (replace this with the actual gn_bf logic)
    processed_frame = gn_sanitize_img(frame_to_tensor(frame))
    processed_frame = edge_sanitize_img(processed_frame)
    processed_frame = tensor_to_frame(processed_frame)
    return processed_frame

def process_video(input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file

    # Create a VideoWriter object to save the processed video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()  # Read a frame from the input video

        if not ret:
            break  # Break the loop if no more frames are available

        # Apply gn_bf() function to process the frame
        processed_frame = gn_bf(frame)


        # Write the processed frame to the output video
        out.write(processed_frame)

    # Release the video objects
    cap.release()
    out.release()
    print(f"Processed video saved to {output_video_path}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='myfriends')
    parser.add_argument("--img_ids", type=str, nargs='+', default=['amy', 'kiat', 'qian', 'yuexin', 'ziyi'])
    parser.add_argument("--attacks", type=str, nargs='+', default=['metacloak', 'aspl', 'glaze', 'mist'])


    args = parser.parse_args()

    for img_id in args.img_ids:
        for attack in args.attacks:
            img_path = f'dataset/{args.dataset_name}/{img_id}_{attack}'
            cleaning(img_path)


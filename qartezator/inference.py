#!/usr/bin/env python3

import argparse
from pathlib import Path
from time import time

import numpy as np
import torch
from PIL import Image


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert ndarray to tensor format.
    Args:
        image (np.ndarray): Image to be converted tensor format
    Returns:
        torch.Tensor
    """
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)
    image = torch.unsqueeze(image, 0)
    return image.float()


def preprocess(image: Image.Image) -> torch.Tensor:
    """Preprocess the input image and mask.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        torch.Tensor
    """
    image = image.convert('RGB')
    w, h = image.size
    # Apply reflection padding sides divisible by 8
    pad_out_to_modulo = 8
    padding_mode = "reflect"
    padded_h = (h // pad_out_to_modulo + 1) * pad_out_to_modulo if h % pad_out_to_modulo != 0 else h
    padded_w = (w // pad_out_to_modulo + 1) * pad_out_to_modulo if w % pad_out_to_modulo != 0 else w
    image = np.pad(image, ((0, padded_h - h), (0, padded_w - w), (0, 0)), mode=padding_mode)
    # Convert to tensor
    image = to_tensor(image / 255.0)
    return image


def process(model: torch.nn.Module, image: Image.Image, device: torch.device) -> Image.Image:
    """Process given image.

    Args:
        model (torch.nn.Module): Loaded model.
        image (PIL.Image.Image): Input image.
        device (torch.device): Device of the model.

    Returns:
        PIL.Image.Image:
    """
    # Resize
    w, h = image.size
    max_side = 600
    if max(h, w) > max_side:
        resize_ratio = max_side / max(h, w)
        w, h = int(resize_ratio * w), int(resize_ratio * h)
        image = image.resize((w, h), Image.BICUBIC)

    image = preprocess(image)

    image = image.to(device)

    with torch.no_grad():
        result = model(image)

    result = result.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    result = np.uint8(255.0 * result)
    result = Image.fromarray(result)
    return result


class Qartezator:
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

    def __call__(self, image: Image.Image) -> Image.Image:
        return process(self.model, image, self.device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-path", type=Path, help="Path to original image.", required=True)
    parser.add_argument("-m", "--model-path", type=Path, help="Path to model.", required=True)
    parser.add_argument("-o", "--output-path", type=Path, help="Output path.", required=True)
    return parser.parse_args()


def main():
    args = get_args()

    image = Image.open(args.image_path)

    lama_model = Qartezator(args.model_path, device="cpu")
    start = time()

    result = lama_model(image)
    print("Time elapsed:", time() - start)
    result.save(args.output_path)


if __name__ == '__main__':
    main()

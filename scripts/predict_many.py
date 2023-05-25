import argparse
import os
from glob import glob
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=str, required=True, help='Source dir.')
    parser.add_argument('--model-path', type=str, required=True, help='Traced model path.')
    parser.add_argument('--output-dir', type=str, required=True, help='Output dir.')
    parser.add_argument('--input-range', type=str, required=False, default='sigmoid', help='Input range.')
    parser.add_argument('--pad-to-modulo', type=int, required=False, default=32, help='Which int multiple?.')
    parser.add_argument('--device', type=str, required=False, default='cpu', help='Which device to use?.')
    return parser.parse_args()


def process(
    model: torch.nn.Module,
    image: Image.Image,
    pad_out_to_modulo=32,
    input_range='sigmoid',
    device='cpu'
) -> Image.Image:
    """Process given image.
​
    Args:
        model (torch.nn.Module): Loaded model.
        image (PIL.Image.Image): Input image.
        pad_out_to_modulo (int): What integer multiple should image width and height be.
​       input_range (Tuple[int, int]): Min and max values of input image.
        device (str): Device to run inference on.

    Returns:
        PIL.Image.Image:
    """
    w, h = image.size
    max_side = 1024
    if max(h, w) > max_side:
        resize_ratio = max_side / max(h, w)
        w, h = int(resize_ratio * w), int(resize_ratio * h)
        image = image.resize((w, h), Image.BICUBIC)
    image = image.convert('RGB')
    padding_mode = 'reflect'
    padded_h = (h // pad_out_to_modulo + 1) * pad_out_to_modulo if h % pad_out_to_modulo != 0 else h
    padded_w = (w // pad_out_to_modulo + 1) * pad_out_to_modulo if w % pad_out_to_modulo != 0 else w
    image = np.pad(image, ((0, padded_h - h), (0, padded_w - w), (0, 0)), mode=padding_mode)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    image = image / 255.0
    if input_range == 'tanh':
        image = (image - 0.5) / 0.5
    image = image.to(device)
    with torch.no_grad():
        result = model(image)
    result = result.squeeze().permute(1, 2, 0).clip(0, 1).detach().cpu().numpy()
    result = np.uint8(255.0 * result)
    result = result[:h, :w]
    result = Image.fromarray(result)
    return result


def main():
    args = get_args()
    model_path = args.model_path
    source_dir = args.source_dir
    output_dir = Path(args.output_dir)
    device = args.device
    pad_out_to_modulo = args.pad_to_modulo
    input_range = args.input_range

    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    output_dir.mkdir(exist_ok=True, parents=True)

    img_extensions = {'.jpg', '.jpeg', '.png'}
    img_paths = []
    for img_extension in img_extensions:
        img_paths += glob(os.path.join(source_dir, '**', f'*{img_extension}'), recursive=True)

    img_paths = sorted(img_paths)
    print(f'Qartezating on {len(img_paths)} images.')

    for img_path in tqdm(img_paths):
        img_name = Path(img_path).name
        img = Image.open(img_path)
        result = process(model, img, pad_out_to_modulo, input_range, device)
        result.save(output_dir / img_name)


if __name__ == '__main__':
    main()

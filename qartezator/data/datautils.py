import random
from pathlib import Path
from typing import Union

import cv2
import numpy as np


def load_image(path: Union[str, Path] = None) -> np.ndarray:
    """Loads RGB image.
    Args:
        path (Union[str, Path]): Path to image.
    Returns:
        np.ndarray.
    """
    image = cv2.imread(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def seed_worker(worker_id):
    """Seeds Dataloader worker"""
    worker_seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    height, width, channels = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, out_height - height), (0, out_width - width), (0, 0)), mode='symmetric')

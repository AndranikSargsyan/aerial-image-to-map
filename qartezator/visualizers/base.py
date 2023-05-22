import abc
from typing import Dict, List

import numpy as np
import torch
from skimage import color


class BaseVisualizer:
    @abc.abstractmethod
    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        """
        Take a batch, make an image from it and visualize
        """
        raise NotImplementedError()


def visualize_images(images_dict: Dict[str, np.ndarray], keys: List[str], rescale_keys=None) -> np.ndarray:
    result = []
    for i, k in enumerate(keys):
        img = images_dict[k]
        img = np.transpose(img, (1, 2, 0))

        if rescale_keys is not None and k in rescale_keys:
            img = img - img.min()
            img /= img.max() + 1e-5
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        result.append(img)
    return np.concatenate(result, axis=1)


def visualize_images_batch(batch: Dict[str, torch.Tensor], keys: List[str], max_items=10,
                                    rescale_keys=None) -> np.ndarray:
    batch = {k: tens.detach().cpu().numpy() for k, tens in batch.items() if k in keys}
    batch_size = next(iter(batch.values())).shape[0]
    items_to_vis = min(batch_size, max_items)
    result = []
    for i in range(items_to_vis):
        cur_dct = {k: tens[i] for k, tens in batch.items()}
        result.append(visualize_images(cur_dct, keys, rescale_keys=rescale_keys))
    return np.concatenate(result, axis=0)

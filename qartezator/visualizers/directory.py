import os

import cv2
import numpy as np

from qartezator.visualizers.base import BaseVisualizer, visualize_images_batch
from qartezator.utils.lamautils import check_and_warn_input_range


class DirectoryVisualizer(BaseVisualizer):
    DEFAULT_KEY_ORDER = 'source_img target_img predicted_img'.split(' ')

    def __init__(self, outdir, key_order=DEFAULT_KEY_ORDER, max_items_in_batch=10, rescale_keys=None):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.key_order = key_order
        self.max_items_in_batch = max_items_in_batch
        self.rescale_keys = rescale_keys

    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        check_and_warn_input_range(batch['source_img'], 0, 1, 'DirectoryVisualizer target image')
        vis_img = visualize_images_batch(batch, self.key_order, max_items=self.max_items_in_batch,
                                                  rescale_keys=self.rescale_keys)

        vis_img = np.clip(vis_img * 255, 0, 255).astype('uint8')

        curoutdir = os.path.join(self.outdir, f'epoch{epoch_i:04d}{suffix}')
        os.makedirs(curoutdir, exist_ok=True)
        rank_suffix = f'_r{rank}' if rank is not None else ''
        out_fname = os.path.join(curoutdir, f'batch{batch_i:07d}{rank_suffix}.jpg')

        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_fname, vis_img)

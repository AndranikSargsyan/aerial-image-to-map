from typing import List

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from qartezator.data.typing import TransformType


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ADDITIONAL_TARGETS = {'target': 'image'}


def get_common_augmentations(crop_size=256):
    transforms = A.Compose([
        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT),
        A.RandomCrop(height=crop_size, width=crop_size, always_apply=True)
    ], additional_targets=ADDITIONAL_TARGETS)
    return transforms


def get_source_augmentations():
    transforms = A.Compose([
        A.RandomBrightnessContrast(),
        A.ISONoise()
    ])
    return transforms


def get_transforms(
    mean: List[float] = None,
    std: List[float] = None,
    augmentations: TransformType = None
) -> TransformType:
    """Creates default base transform.
    Args:
        mean (List[float]): List of means for normalization.
        std (List[float]): List of stds for normalization.
        augmentations (TransformType): Augmentations to use.
    Returns:
        TransformType
    """
    transforms_list = []
    if augmentations is not None:
        transforms_list.append(augmentations)
    if mean is not None and std is not None:
        transforms_list.append(A.Normalize(mean=mean, std=std))
    transforms_list.append(ToTensorV2())
    transform = A.Compose(transforms_list, additional_targets=ADDITIONAL_TARGETS)
    return transform

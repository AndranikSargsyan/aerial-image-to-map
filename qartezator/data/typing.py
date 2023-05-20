from typing import Union

import albumentations as A
import numpy as np
import torch

TransformType = Union[A.BasicTransform, A.BaseCompose]
DatasetElement = Union[np.ndarray, torch.Tensor]

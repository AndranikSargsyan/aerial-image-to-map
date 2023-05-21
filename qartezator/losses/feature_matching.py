from typing import List

import torch
import torch.nn.functional as F


def l2_loss(pred, target):
    per_pixel_l2 = F.mse_loss(pred, target, reduction='none')
    return per_pixel_l2.mean()


def l1_loss(pred, target):
    per_pixel_l1 = F.l1_loss(pred, target, reduction='none')
    return per_pixel_l1.mean()


def feature_matching_loss(fake_features: List[torch.Tensor], target_features: List[torch.Tensor]):
    res = torch.stack([F.mse_loss(fake_feat, target_feat)
                       for fake_feat, target_feat in zip(fake_features, target_features)]).mean()
    return res

import torch

from qartezator.evaluation.evaluator import EvaluatorOnline
from qartezator.evaluation.metrics.base import SSIMScore, LPIPSScore, FIDScore


def make_evaluator(ssim=True, lpips=True, fid=True, **kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = {}
    if ssim:
        metrics['ssim'] = SSIMScore()
    if lpips:
        metrics['lpips'] = LPIPSScore()
    if fid:
        metrics['fid'] = FIDScore().to(device)
    return EvaluatorOnline(scores=metrics, **kwargs)

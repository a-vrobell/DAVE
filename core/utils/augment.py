import torch
import torch.nn as nn
from torch import Tensor
import kornia.augmentation as KA

from typing import Tuple
from dataclasses import dataclass


@dataclass
class AugCFG:
    h_flip_prob: float
    affine_prob: float
    noise_alpha: float
    
    rotate_range:    Tuple[float, float]
    translate_range: Tuple[float, float]
    scale_range:     Tuple[float, float]


class DiffAugment(nn.Module):
    def __init__(self, cfg: AugCFG):
        super().__init__()
        self.cfg = cfg
        self.noise_alpha = cfg.noise_alpha

        self.geometric = KA.AugmentationSequential(
            KA.RandomHorizontalFlip(p=cfg.h_flip_prob),
            KA.RandomAffine(
                degrees=cfg.rotate_range,
                translate=cfg.translate_range,
                scale=cfg.scale_range,
                p=cfg.affine_prob,
                shear=None,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        torch.seed()
        return self.geometric(x)

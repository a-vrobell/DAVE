import torch
from torch import Tensor

from pathlib import Path
from typing import List, Any

from core.config import DAVEConfig
from core.utils.detach_mode import (
    detach_gelu,
    detach_attention,
    detach_layer_norm,
    attach_gelu,
    attach_attention,
    attach_layer_norm,   
)


class DAVEExplainer:
    def __init__(
        self, 
        model_cfg_path: Path,
        device: torch.device,
        eps: float = 1e-8
    ):
        self.eps = eps
        self._op_variation_removed: bool = False
        
        cfg = DAVEConfig.load_from_yaml(
            path=model_cfg_path, 
            eps=eps,
        )

        self.model_name = cfg.model_name
        self.model = cfg.model.to(device)
        self.aug = cfg.aug.to(device)
        self.post_proc = cfg.post_proc.to(device)
        self.input_transform = cfg.input_transform

    def explain(
        self,
        x: Tensor,
        y: Tensor,
        num_steps: int,
        post_proc: bool = True,
    ) -> Tensor:
        """
        Performs DAVE attribution.
        """
        self.model.eval()
        self.aug.train()
        self.remove_operator_variation()

        t_schedule = self.get_noise_schedule(
            num_steps=num_steps,
        ).to(x.device)

        buffer = []
        for step_idx in range(num_steps):
            t = t_schedule[step_idx]
            c = self.effective_transform(x=x, y=y, t=t)
            c = c.detach() * x.detach()
            buffer.append(c)

        c = self._aggregate_maps(buffer)

        if post_proc:
            c = self.post_proc(c)

        self.restore_operator_variation()
        return c

    def effective_transform(
        self, x: Tensor, y: Tensor, t: Tensor,
    ) -> Tensor:
        """
        Computes Effective Transformation,
        assuming removed operator variation
        via remove_operator_variation().
        """
        assert self._op_variation_removed, (
            "Call remove_operator_variation() first!"
        )

        x = self._clone_input(x)
        z = self.pred_batch(x, y, t)

        # After calling remove_operator_variation(), 
        # grad becomes effective transform;
        w_eff = torch.autograd.grad(
            outputs=z,
            inputs=[x],
            grad_outputs=torch.ones_like(z),
            retain_graph=False,
        )[0]
        return w_eff.detach()

    def pred_batch(
        self, x: Tensor, y: Tensor, t: Tensor,
    ) -> Tensor:
        """
        Predicts image batch with: 
        - spatial augmentations (for equivariant transform) 
        - noise addition (for low-pass filter). 
        """
        self._check_batch_shapes(x, y, t)

        x = self.aug(x)
        x = self.add_noise(x, t)
        z = self.model(x)

        y = y.unsqueeze(-1).long()
        z = z.gather(dim=1, index=y)
        return z

    def get_noise_schedule(self, num_steps: int) -> Tensor:
        return torch.linspace(
            0.0, self.aug.noise_alpha, steps=num_steps,
        )

    def add_noise(self, x: Tensor, t: Tensor) -> Tensor:
        torch.seed()
        noise = torch.randn_like(x, device=x.device)
        x = (1.0 - t) * x + torch.sqrt(1.0 - (1.0 - t)**2) * noise
        return x

    def remove_operator_variation(self):
        """
        Converts model grad to effective transform.
        """
        detach_gelu(self.model)
        detach_attention(self.model)
        detach_layer_norm(self.model)
        self._op_variation_removed = True

    def restore_operator_variation(self):
        """
        Converts model grad back to gradient.
        """
        attach_gelu(self.model)
        attach_attention(self.model)
        attach_layer_norm(self.model)
        self._op_variation_removed = False

    def _aggregate_maps(self, maps: List[Tensor]) -> Tensor:
        x = torch.stack(maps, dim=0)
        mask = self._mad_mask(x)
        x = (x * mask).sum(dim=0)
        den = mask.sum(dim=0).clamp(min=1)
        return x / den

    def _mad_mask(self, x: Tensor) -> Tensor:
        """
        Median Absolute Deviation mask for outliers.
        """
        med = x.median(dim=0).values
        mad = (x - med).abs().median(dim=0).values
        scale = 1.4826 * mad + self.eps
        mask = (x - med).abs() <= 2.5 * scale
        return mask

    def _clone_input(self, x: Tensor) -> Tensor:
        return x.clone().detach().requires_grad_(True)

    def _check_batch_shapes(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
    ):
        assert x.ndim == 4, "Expected batch of image samples!"
        assert t.ndim == 0, "Expected batch of noise levels!"
        assert y.ndim == 1, "Expected batch of sample labels!"

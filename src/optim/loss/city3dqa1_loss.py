"""Losses for City3DQA1 pretraining without external chamfer ranking files."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from optim.loss.contra_loss import ClipLoss
from optim.loss.loss import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class City3DQA1SceneViewPMLoss(nn.Module):
    """Contrast City3DQA1 point-map features with RGB and text features."""

    def __init__(self, cfg, accelerator):
        super().__init__()
        self.accelerator = accelerator
        self.contrast_loss = ClipLoss(
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )

    def forward(self, data_dict):
        logit_scale = data_dict["logit_scale"]
        view_rgb = F.normalize(data_dict["inter_view_rgb_embed"], p=2, dim=-1)
        view_txt = F.normalize(data_dict["inter_view_txt_embed"], p=2, dim=-1)
        view_pm = F.normalize(data_dict["inter_view_pm_embed"], p=2, dim=-1)
        view_ground_txt = F.normalize(data_dict["inter_view_ground_txt_embed"], p=2, dim=-1)

        scene_rgb = F.normalize(data_dict["scene_rgb_embed"], p=2, dim=-1)
        scene_pm = F.normalize(data_dict["scene_pm_embed"], p=2, dim=-1)

        batch_size, num_views, dim = view_pm.shape
        view_pm_f = view_pm.reshape(-1, dim)
        view_rgb_f = view_rgb.reshape(-1, dim)
        view_txt_f = view_txt.reshape(-1, dim)

        losses = {
            "loss_rgb_pm_view": self.contrast_loss(view_pm_f, view_rgb_f, logit_scale),
            "loss_txt_pm_view": self.contrast_loss(view_pm_f, view_txt_f, logit_scale),
            "loss_rgb_pm_scene": self.contrast_loss(scene_pm, scene_rgb, logit_scale),
        }

        if "scene_text_embed" in data_dict:
            scene_txt = F.normalize(data_dict["scene_text_embed"], p=2, dim=-1)
            losses["loss_txt_pm_scene"] = self.contrast_loss(scene_pm, scene_txt, logit_scale)

        gt_logits = torch.bmm(view_pm, view_ground_txt.transpose(1, 2)) * logit_scale
        targets = torch.arange(num_views, device=gt_logits.device).unsqueeze(0).expand(batch_size, num_views)
        losses["loss_grounded_view"] = (
            F.cross_entropy(gt_logits.reshape(batch_size * num_views, num_views), targets.reshape(-1))
            + F.cross_entropy(gt_logits.transpose(1, 2).reshape(batch_size * num_views, num_views), targets.reshape(-1))
        ) * 0.5

        return losses

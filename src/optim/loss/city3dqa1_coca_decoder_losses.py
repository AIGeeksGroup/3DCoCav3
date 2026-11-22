"""Losses for City3DQA1 CoCa decoder text generation."""

import torch.nn as nn
import torch.nn.functional as F

from optim.loss.loss import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class City3DQA1CoCaDecoderGenerationLoss(nn.Module):
    """Token-level teacher-forcing loss for caption and QA generation."""

    def __init__(self, cfg, accelerator):
        super().__init__()
        self.ignore_index = int(cfg.model.get("generation", {}).get("ignore_index", -100))

    def forward(self, data_dict):
        logits = data_dict["text_logits"]
        labels = data_dict["decoder_labels"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
        )
        return {"city3dqa1_coca_decoder_generation_loss": loss}


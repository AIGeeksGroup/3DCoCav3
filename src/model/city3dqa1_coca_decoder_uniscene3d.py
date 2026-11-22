"""City3DQA1 UniScene3D models with a CoCa-style generative text decoder."""

from pathlib import Path

import torch
from torch.amp import autocast
from transformers import AutoTokenizer

from common.misc import build_fgclip_model_from_local_code_with_hf_weights
from model.build import BaseModel, MODEL_REGISTRY
from modules.build import build_module
from optim.utils import no_decay_param_group


class _BaseCity3DQA1UrbanCoCaDecoderPTUniScene3D(BaseModel):
    """Encode cached RGB/point-map views and decode target text autoregressively."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        model_root = str(Path(__file__).resolve().parents[1] / "fg-clip")
        fgclip_repo_id = self.cfg.model.get("fgclip_repo_id", None)
        if fgclip_repo_id is None:
            self.pm_encoder = build_fgclip_model_from_local_code_with_hf_weights(model_root)
        else:
            self.pm_encoder = build_fgclip_model_from_local_code_with_hf_weights(model_root, repo_id=fgclip_repo_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model_root)
        self.head_list = self.cfg.model.heads.head_list
        for head in self.head_list:
            setattr(self, head, build_module("heads", getattr(self.cfg.model.heads, head)))

        self._decoder_head_name = self.head_list[0]
        self.set_downstream_mode()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def set_downstream_mode(self):
        if not self.cfg.model.get("train_pm_encoder", False):
            self.freeze_module(self.pm_encoder)
            self.pm_encoder.eval()
        else:
            self.pm_encoder.train()

        for head in self.head_list:
            getattr(self, head).train()

    def _encode_scene_tokens(self, data_dict):
        images = data_dict["images"]
        point_maps = data_dict["point_map"]
        if images.ndim != 5 or point_maps.ndim != 5:
            raise ValueError("Expected images and point_map with shape [B, V, C, H, W].")

        batch_size, num_views, channels, height, width = images.shape
        if point_maps.shape[:2] != (batch_size, num_views):
            raise ValueError("images and point_map must have matching batch and view dimensions.")

        color_pm = torch.cat(
            [
                images.reshape(batch_size * num_views, channels, height, width).contiguous().float(),
                point_maps.reshape(batch_size * num_views, point_maps.shape[2], height, width).contiguous().float(),
            ],
            dim=1,
        )

        device_type = "cuda" if color_pm.is_cuda else "cpu"
        autocast_enabled = device_type == "cuda"
        grad_enabled = bool(self.cfg.model.get("train_pm_encoder", False))

        with torch.set_grad_enabled(grad_enabled):
            with autocast(device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
                _, vision_feat = self.pm_encoder.get_image_features(color_pm)

        return vision_feat.reshape(batch_size, num_views, -1)

    def forward(self, data_dict, mode=None):
        scene_tokens = self._encode_scene_tokens(data_dict)
        data_dict["inter_view_pm_embed"] = scene_tokens
        data_dict["scene_pm_embed"] = scene_tokens.mean(dim=1)

        question_input_ids = data_dict.get("question_input_ids", None)
        question_attention_mask = data_dict.get("question_attention_mask", None)
        view_attention_mask = data_dict.get("view_attention_mask", None)
        view_ids = data_dict.get("view_ids", None)

        decoder_head = getattr(self, self._decoder_head_name)
        logits = decoder_head(
            scene_tokens=scene_tokens,
            decoder_input_ids=data_dict["decoder_input_ids"],
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask,
            scene_attention_mask=view_attention_mask,
            view_ids=view_ids,
        )
        data_dict["text_logits"] = logits

        if not self.training:
            max_new_tokens = int(self.cfg.model.get("generation", {}).get("max_new_tokens", 32))
            generated_ids = decoder_head.generate(
                scene_tokens=scene_tokens,
                question_input_ids=question_input_ids,
                question_attention_mask=question_attention_mask,
                scene_attention_mask=view_attention_mask,
                view_ids=view_ids,
                max_new_tokens=max_new_tokens,
            )
            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            if "sentence" in data_dict:
                data_dict["answer_pred"] = decoded
            else:
                data_dict["caption_pred"] = decoded

        return data_dict

    def get_opt_params(self):
        def get_lr(cfg, default_lr):
            return default_lr if cfg.get("lr") is None else cfg.get("lr")

        optimizer_grouped_parameters = []
        if self.cfg.model.get("train_pm_encoder", False):
            optimizer_grouped_parameters += no_decay_param_group(
                self.pm_encoder.named_parameters(),
                get_lr(self.cfg.model.vision, self.cfg.solver.lr),
            )

        for head in self.head_list:
            head_cfg = getattr(self.cfg.model.heads, head)
            optimizer_grouped_parameters += no_decay_param_group(
                getattr(self, head).named_parameters(),
                get_lr(head_cfg, self.cfg.solver.lr),
            )
        return optimizer_grouped_parameters


@MODEL_REGISTRY.register()
class City3DQA1CaptionUrbanCoCaDecoderPTUniScene3D(_BaseCity3DQA1UrbanCoCaDecoderPTUniScene3D):
    """Caption-only City3DQA1 model using PT visual cache and CoCa decoder."""


@MODEL_REGISTRY.register()
class City3DQA1QAUrbanCoCaDecoderPTUniScene3D(_BaseCity3DQA1UrbanCoCaDecoderPTUniScene3D):
    """Generative QA City3DQA1 model using PT visual cache and CoCa decoder."""

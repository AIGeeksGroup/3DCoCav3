"""City3DQA1-specific UniScene3D model without debug breakpoints."""

import math
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.amp import autocast
from transformers import AutoModel
from transformers import AutoTokenizer

from model.build import MODEL_REGISTRY
from model.uniscene3d import UniScene3D


class _City3DQA1DummyTokenizer:
    """Placeholder for Jina's optional internal encode() tokenizer."""

    init_kwargs = {"model_max_length": 8192}

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "City3DQA1QAUniScene3D does not use Jina's internal encode() tokenizer. "
            "Questions are tokenized by UniScene3D.tokenizer before text_model forward."
        )


@contextmanager
def _city3dqa1_skip_jina_internal_tokenizer():
    """Avoid missing Jina tokenizer assets during City3DQA1 QA construction."""
    original_model_descriptor = AutoModel.__dict__["from_pretrained"]
    original_model_from_pretrained = AutoModel.from_pretrained
    original_descriptor = AutoTokenizer.__dict__["from_pretrained"]
    original_from_pretrained = AutoTokenizer.from_pretrained
    fgclip_root = str(Path(__file__).resolve().parents[1] / "fg-clip")

    def patched_model_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        if str(pretrained_model_name_or_path) == "jinaai/jina-clip-v2":
            kwargs.setdefault("use_text_flash_attn", False)
        return original_model_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def patched_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        model_name = str(pretrained_model_name_or_path)
        if model_name == "jinaai/jina-embeddings-v3":
            return _City3DQA1DummyTokenizer()
        if model_name == "jinaai/jina-clip-v2":
            kwargs.pop("trust_remote_code", None)
            return original_from_pretrained(fgclip_root, *args, **kwargs)
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    AutoModel.from_pretrained = staticmethod(patched_model_from_pretrained)
    AutoTokenizer.from_pretrained = staticmethod(patched_from_pretrained)
    try:
        yield
    finally:
        AutoModel.from_pretrained = original_model_descriptor
        AutoTokenizer.from_pretrained = original_descriptor


@MODEL_REGISTRY.register()
class City3DQA1UniScene3D(UniScene3D):
    """UniScene3D pretraining model for City3DQA1 rendered urban scenes."""

    _trainable_pretrain_keywords = (
        "vision_model.",
        "visual_projection",
    )

    def set_pm_encoder_trainability(self, train_geo_embed_only=False):
        """Freeze unused FG-CLIP branches so DDP only tracks active parameters."""
        self.freeze_module(self.pm_encoder)
        for name, param in self.pm_encoder.named_parameters():
            if any(keyword in name for keyword in self._trainable_pretrain_keywords):
                param.requires_grad = True

    def get_pretrain_params(self):
        """Return only the City3DQA1 pretraining parameters used by this forward pass."""
        return [
            (name, param)
            for name, param in self.pm_encoder.named_parameters()
            if param.requires_grad
        ]

    def augment_pretrain_pointmaps(self, point_maps):
        aug_cfg = self.cfg.model.get("pointmap_augmentation", {})
        if (not self.training) or (not aug_cfg.get("enabled", False)):
            return point_maps

        rotations_deg = aug_cfg.get("z_rotations_deg", [90, 180, 270])
        if len(rotations_deg) == 0:
            return point_maps

        scale_min, scale_max = aug_cfg.get("scale_range", [0.75, 1.25])
        translate_min, translate_max = aug_cfg.get("translation_range", [-1.0, 1.0])

        batch_size, num_views, height, width, channels = point_maps.shape
        if channels != 3:
            raise ValueError(f"Expected point maps with 3 channels, got {channels}.")

        points = point_maps.float().reshape(batch_size, -1, channels)
        valid_mask = torch.isfinite(points).all(dim=-1) & (points.abs().sum(dim=-1) > 0)
        valid_points = torch.where(valid_mask.unsqueeze(-1), points, torch.zeros_like(points))
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        centroids = valid_points.sum(dim=1) / valid_counts.float()

        rotation_choices = torch.tensor(rotations_deg, device=points.device, dtype=points.dtype)
        rotation_indices = torch.randint(0, rotation_choices.numel(), (batch_size,), device=points.device)
        angles = rotation_choices[rotation_indices] * (math.pi / 180.0)
        cos_theta = torch.cos(angles)
        sin_theta = torch.sin(angles)

        rotation_mats = torch.zeros(batch_size, 3, 3, device=points.device, dtype=points.dtype)
        rotation_mats[:, 0, 0] = cos_theta
        rotation_mats[:, 0, 1] = -sin_theta
        rotation_mats[:, 1, 0] = sin_theta
        rotation_mats[:, 1, 1] = cos_theta
        rotation_mats[:, 2, 2] = 1.0

        scales = torch.empty(batch_size, 1, 1, device=points.device, dtype=points.dtype).uniform_(
            float(scale_min), float(scale_max)
        )
        translations = torch.empty(batch_size, 1, 3, device=points.device, dtype=points.dtype).uniform_(
            float(translate_min), float(translate_max)
        )

        centered = points - centroids.unsqueeze(1)
        rotated = torch.einsum("bnc,bdc->bnd", centered, rotation_mats)
        augmented_points = rotated * scales + centroids.unsqueeze(1) + translations
        points = torch.where(valid_mask.unsqueeze(-1), augmented_points, points)
        return points.reshape(batch_size, num_views, height, width, channels).to(point_maps.dtype)

    def forward(self, data_dict, mode=None):
        if "cur_step" not in data_dict:
            data_dict["cur_step"] = 1
            data_dict["total_steps"] = 1

        data_dict["logit_scale"] = self.pm_encoder.logit_scale.exp()

        if mode != "pretrain":
            return super().forward(data_dict, mode=mode)

        batch_size, num_views, _, _, _ = data_dict["point_map"].shape
        data_dict["point_map"] = self.augment_pretrain_pointmaps(
            data_dict["point_map"].to(torch.float32, non_blocking=True)
        )
        data_dict["point_map"] = data_dict["point_map"].to(torch.bfloat16, non_blocking=True).permute(0, 1, 4, 2, 3)

        dropout_stats = self.apply_pretrain_modality_dropout(data_dict["images"], data_dict["point_map"])
        if dropout_stats is not None:
            data_dict["drop_rgb_count"] = dropout_stats["drop_rgb_count"]
            data_dict["drop_pointmap_count"] = dropout_stats["drop_pointmap_count"]
            data_dict["keep_both_count"] = dropout_stats["keep_both_count"]
            data_dict["has_pointmap_input"] = dropout_stats["use_geo_embedding"]
        else:
            data_dict["has_pointmap_input"] = torch.ones(
                batch_size, dtype=torch.bool, device=data_dict["point_map"].device
            )

        pm_basic_features = []
        device_type = "cuda" if data_dict["point_map"].is_cuda else "cpu"
        autocast_enabled = device_type == "cuda"

        for i in range(batch_size):
            with autocast(device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
                use_patch_embedding = True
                use_geo_embedding = True
                if dropout_stats is not None:
                    use_patch_embedding = bool(dropout_stats["use_patch_embedding"][i].item())
                    use_geo_embedding = bool(dropout_stats["use_geo_embedding"][i].item())

                color_pm = torch.cat([data_dict["images"][i], data_dict["point_map"][i]], dim=1)
                _, pm_feat = self.pm_encoder.get_image_features(
                    color_pm,
                    use_patch_embedding=use_patch_embedding,
                    use_geo_embedding=use_geo_embedding,
                )
                pm_basic_features.append(pm_feat)

        data_dict["inter_view_pm_embed"] = torch.stack(pm_basic_features, dim=0)
        data_dict["scene_pm_embed"] = data_dict["inter_view_pm_embed"].mean(dim=1)

        lang_basic_features = []
        ground_lang_basic_features = []
        rgb_basic_features = []

        with torch.no_grad():
            with autocast(device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
                for i in range(batch_size):
                    lang_basic_features.append(
                        self.frozen_model.get_text_features(data_dict["txt_ids"][i], walk_short_pos=True)
                    )
                    ground_lang_basic_features.append(
                        self.frozen_model.get_text_features(data_dict["ground_txt_ids"][i], walk_short_pos=True)
                    )
                    rgb_basic_features.append(self.frozen_model.get_image_features(data_dict["images"][i])[1])

                if getattr(self, "use_scene_cap", False):
                    data_dict["scene_text_embed"] = self.frozen_model.get_text_features(
                        data_dict["scene_txt_ids"], walk_short_pos=False
                    )

        data_dict["inter_view_txt_embed"] = torch.stack(lang_basic_features, dim=0)
        data_dict["inter_view_ground_txt_embed"] = torch.stack(ground_lang_basic_features, dim=0)
        data_dict["inter_view_rgb_embed"] = torch.stack(rgb_basic_features, dim=0)
        data_dict["scene_rgb_embed"] = data_dict["inter_view_rgb_embed"].mean(dim=1)
        return data_dict


@MODEL_REGISTRY.register()
class City3DQA1QAUniScene3D(City3DQA1UniScene3D):
    """City3DQA1 QA fine-tuning model with Jina tokenizer-cache workaround."""

    def __init__(self, cfg):
        with _city3dqa1_skip_jina_internal_tokenizer():
            super().__init__(cfg)

"""Generation-only City3DQA1 model with a frozen Pointmap encoder."""

import torch

from model.build import MODEL_REGISTRY
from model.city3dqa1_coca_decoder_uniscene3d import (
    _BaseCity3DQA1UrbanCoCaDecoderPTUniScene3D,
)


@MODEL_REGISTRY.register()
class City3DQA1MSGCaptionGenerationFrozenPM(
    _BaseCity3DQA1UrbanCoCaDecoderPTUniScene3D
):
    """Train only the CoCa decoder heads.

    The Pointmap/RGB encoder is frozen by the base class.  There are no
    alignment projections or contrastive teacher branches in this model, so
    generation loss is the only source of updates and only decoder-head
    parameters are returned by ``get_opt_params``.
    """

    @staticmethod
    def _point_map_channels_first(point_maps):
        if point_maps.ndim != 5:
            raise ValueError("Expected point_map with shape [B,V,H,W,3] or [B,V,3,H,W].")
        if point_maps.shape[-1] == 3:
            return point_maps.permute(0, 1, 4, 2, 3).contiguous()
        if point_maps.shape[2] == 3:
            return point_maps.contiguous()
        raise ValueError("point_map must have three coordinate channels.")

    @staticmethod
    def _autocast_context(tensor):
        from torch.amp import autocast

        return autocast(
            tensor.device.type,
            dtype=torch.bfloat16,
            enabled=tensor.device.type == "cuda",
        )

    def _encode_scene_tokens(self, data_dict):
        images = data_dict["images"]
        point_maps = self._point_map_channels_first(data_dict["point_map"])
        if images.ndim != 5 or point_maps.ndim != 5:
            raise ValueError("Expected images and point_map with shape [B,V,C,H,W].")

        batch_size, num_views, channels, height, width = images.shape
        flat_images = images.reshape(
            batch_size * num_views, channels, height, width
        ).float()
        flat_point_maps = point_maps.reshape(
            batch_size * num_views,
            point_maps.shape[2],
            point_maps.shape[3],
            point_maps.shape[4],
        ).float()
        color_pm = torch.cat([flat_images, flat_point_maps], dim=1)

        with torch.no_grad():
            with self._autocast_context(color_pm):
                _, vision_features = self.pm_encoder.get_image_features(color_pm)

        scene_tokens = vision_features.reshape(batch_size, num_views, -1)
        data_dict["inter_view_pm_embed"] = scene_tokens
        data_dict["scene_pm_embed"] = scene_tokens.mean(dim=1)
        return scene_tokens

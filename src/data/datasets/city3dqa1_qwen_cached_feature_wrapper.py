"""Qwen wrapper that reads precomputed FG-CLIP scene-view features."""

from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .dataset_wrapper import DATASETWRAPPER_REGISTRY


DEFAULT_FEATURE_ROOT = Path(
    "/data1/znn/Dataset_pk/CIty-3DQA/FG_CLIP_Feature/"
    "1scene_sameview_sce_caption_ckpt_6000"
)


@DATASETWRAPPER_REGISTRY.register()
class City3DQA1QwenCachedFeatureMixedWrapper(Dataset):
    """Align each scene's saved features to source view IDs before sampling."""

    def __init__(self, cfg, dataset, split="train"):
        self.cfg = cfg
        self.dataset = dataset
        self.split = split
        self.num_views = int(cfg.get("num_views", 50))
        self.feature_root = Path(cfg.model.get("feature_root", DEFAULT_FEATURE_ROOT))
        if not self.feature_root.is_dir():
            raise FileNotFoundError(f"FG-CLIP feature directory not found: {self.feature_root}")

        model_path = cfg.model.get("qwen_model_path")
        if not model_path:
            model_path = str(Path(__file__).resolve().parents[3] / "pretrained" / "Qwen3.8-2B-Distill")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.qa_max_target_length = int(
            cfg.data.City3DQA1QAUrbanCoCaDecoderPT.get("max_target_length", 64)
        )
        self.caption_max_target_length = int(
            cfg.data.City3DQA1CaptionUrbanCoCaDecoderPT.get("max_target_length", 512)
        )
        self.feature_paths = self._discover_feature_paths()
        self._feature_cache = {}
        self._feature_cache_order = []
        self.feature_cache_size = int(cfg.model.get("feature_scene_cache_size", 8))

    def _discover_feature_paths(self):
        paths = {}
        for city_dir in sorted(path for path in self.feature_root.iterdir() if path.is_dir()):
            for path in sorted(city_dir.glob("*.pt")):
                scene_id = f"{city_dir.name}_{path.stem}"
                if scene_id in paths:
                    raise ValueError(f"Duplicate feature scene_id {scene_id} under {self.feature_root}")
                paths[scene_id] = path
        if not paths:
            raise FileNotFoundError(f"No City/Area feature files found under {self.feature_root}")
        return paths

    def __len__(self):
        return len(self.dataset)

    def _load_scene_features(self, scene_id):
        if scene_id in self._feature_cache:
            self._feature_cache_order.remove(scene_id)
            self._feature_cache_order.append(scene_id)
            return self._feature_cache[scene_id]
        if scene_id not in self.feature_paths:
            raise KeyError(f"No FG-CLIP features found for scene_id={scene_id}")

        path = self.feature_paths[scene_id]
        feature_scene = torch.load(path, map_location="cpu", weights_only=True)
        if feature_scene.get("scene_id") != scene_id:
            raise ValueError(
                f"Feature scene_id mismatch in {path}: "
                f"expected {scene_id}, found {feature_scene.get('scene_id')}"
            )
        features = feature_scene.get("features")
        names = feature_scene.get("file_names")
        view_ids = feature_scene.get("view_ids")
        if not torch.is_tensor(features) or features.ndim != 2 or features.shape[1] != 512:
            raise ValueError(f"Expected [views,512] features in {path}")
        if not isinstance(names, list) or not torch.is_tensor(view_ids):
            raise ValueError(f"Missing file_names or view_ids in {path}")
        if len(names) != features.shape[0] or view_ids.numel() != features.shape[0]:
            raise ValueError(f"Feature/view metadata length mismatch in {path}")
        if len(set(str(name) for name in names)) != len(names):
            raise ValueError(f"Duplicate file_names in {path}")
        if view_ids.unique().numel() != view_ids.numel():
            raise ValueError(f"Duplicate view_ids in {path}")

        result = {
            "features": features,
            "file_names": [str(name) for name in names],
            "view_ids": view_ids.long(),
        }
        self._feature_cache[scene_id] = result
        self._feature_cache_order.append(scene_id)
        while len(self._feature_cache_order) > self.feature_cache_size:
            old_scene_id = self._feature_cache_order.pop(0)
            self._feature_cache.pop(old_scene_id, None)
        return result

    def _align_scene_features(self, sample):
        scene_id = str(sample.get("scan_id", ""))
        feature_scene = self._load_scene_features(scene_id)
        sample_names = [str(name) for name in sample.get("file_names", [])]
        sample_view_ids = sample.get("view_ids")
        if not torch.is_tensor(sample_view_ids):
            raise ValueError(f"Sample {scene_id} has no tensor view_ids")
        sample_view_ids = sample_view_ids.long().cpu()
        if sample_names != feature_scene["file_names"]:
            raise ValueError(
                f"View filename/order mismatch for {scene_id}: source cache and FG-CLIP features differ"
            )
        if not torch.equal(sample_view_ids, feature_scene["view_ids"]):
            raise ValueError(
                f"view_ids mismatch for {scene_id}: source cache and FG-CLIP features differ"
            )
        return feature_scene["features"]

    def _build_view_indices(self, num_views):
        if num_views == self.num_views:
            return torch.arange(num_views)
        if num_views > self.num_views:
            if self.split == "train":
                return torch.randperm(num_views)[: self.num_views]
            return torch.linspace(0, num_views - 1, self.num_views).long()
        pad_indices = torch.randint(0, num_views, (self.num_views - num_views,))
        return torch.cat([torch.arange(num_views), pad_indices], dim=0)

    def _chat_sequence(self, is_caption, sentence, target, max_length):
        user_content = (
            "Describe the 3D scene in one caption."
            if is_caption
            else f"Answer the question about the 3D scene.\nQuestion: {sentence}"
        )
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        complete = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": str(target)},
            ],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        if complete[: len(prompt)] != prompt:
            raise ValueError("Qwen chat template prompt is not a prefix of the full conversation.")
        eos = self.tokenizer.eos_token_id
        if eos is not None and (not complete or complete[-1] != eos):
            complete.append(eos)
        complete = complete[: max_length + 1]
        prompt_len = min(len(prompt), max_length)
        prompt_ids = torch.tensor(prompt[:max_length], dtype=torch.long)
        ids = torch.tensor(complete, dtype=torch.long)
        decoder_input_ids = ids[:-1].contiguous()
        labels = ids[1:].clone()
        labels[: max(0, prompt_len - 1)] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        return decoder_input_ids, labels, torch.ones_like(decoder_input_ids), prompt_ids

    def __getitem__(self, index):
        sample = self.dataset[index]
        out = {
            key: value.contiguous().clone() if torch.is_tensor(value) else value
            for key, value in sample.items()
        }
        is_caption = "caption_text" in out and "sentence" not in out
        max_length = self.caption_max_target_length if is_caption else self.qa_max_target_length

        scene_features = self._align_scene_features(out)
        indices = self._build_view_indices(scene_features.shape[0])
        out["scene_features"] = scene_features[indices].contiguous()
        if torch.is_tensor(out.get("view_ids")):
            out["view_ids"] = out["view_ids"][indices].contiguous()
        if isinstance(out.get("file_names"), list):
            out["file_names"] = [out["file_names"][int(i)] for i in indices.tolist()]
        out.pop("images", None)
        out.pop("point_map", None)

        target = out.get("target_text", "")
        ids, labels, mask, prompt_ids = self._chat_sequence(
            is_caption, out.get("sentence", ""), target, max_length
        )
        out["decoder_input_ids"] = ids
        out["decoder_labels"] = labels
        out["decoder_attention_mask"] = mask
        out["generation_input_ids"] = prompt_ids
        out["generation_attention_mask"] = torch.ones_like(prompt_ids)
        out["target_text"] = target
        return out

    def collate_fn(self, batch_list):
        if not batch_list:
            raise ValueError("Cannot collate an empty QA/caption batch.")
        pad_id = self.tokenizer.pad_token_id
        sequence_fields = {
            "decoder_input_ids": pad_id,
            "decoder_labels": -100,
            "decoder_attention_mask": 0,
            "generation_input_ids": pad_id,
            "generation_attention_mask": 0,
        }
        collated = {}
        keys = dict.fromkeys(key for sample in batch_list for key in sample)
        for key in keys:
            values = [sample.get(key) for sample in batch_list]
            exemplar = next(value for value in values if value is not None)
            if key in sequence_fields:
                values = [exemplar.new_empty(0) if value is None else value for value in values]
                collated[key] = pad_sequence(
                    values, batch_first=True, padding_value=sequence_fields[key]
                )
            elif torch.is_tensor(exemplar):
                if key == "question_type":
                    values = [torch.full_like(exemplar, -1) if value is None else value for value in values]
                elif key == "scene_caption_valid":
                    values = [torch.zeros_like(exemplar) if value is None else value for value in values]
                elif any(value is None for value in values):
                    raise ValueError(f"Missing required tensor {key} in QA/caption batch.")
                collated[key] = torch.stack(values, dim=0)
            else:
                collated[key] = ["" if value is None else value for value in values]
        return collated

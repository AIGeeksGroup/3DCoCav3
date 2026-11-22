"""Mixed City3DQA1 QA + caption wrapper for CoCa decoder PT training."""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoTokenizer

from .dataset_wrapper import DATASETWRAPPER_REGISTRY


def _build_tokenizer():
    model_root = str(Path(__file__).resolve().parents[2] / "fg-clip")
    return AutoTokenizer.from_pretrained(model_root)


def _build_image_processor():
    model_root = str(Path(__file__).resolve().parents[2] / "fg-clip")
    return AutoImageProcessor.from_pretrained(model_root)


@DATASETWRAPPER_REGISTRY.register()
class City3DQA1QAAndCaptionUrbanCoCaDecoderPTWrapper(Dataset):
    """Wrap QA and caption samples with the right target length per sample type."""

    def __init__(self, cfg, dataset, split="train"):
        self.cfg = cfg
        self.dataset = dataset
        self.split = split
        self.num_views = cfg.get("num_views", 32)
        self.tokenizer = _build_tokenizer()
        self.image_processor = _build_image_processor()

        self.qa_cfg = cfg.data.get("City3DQA1QAUrbanCoCaDecoderPT", {})
        self.caption_cfg = cfg.data.get("City3DQA1CaptionUrbanCoCaDecoderPT", {})
        self.qa_max_target_length = int(self.qa_cfg.get("max_target_length", 77))
        self.qa_max_question_length = int(self.qa_cfg.get("max_question_length", 256))
        self.caption_max_target_length = int(self.caption_cfg.get("max_target_length", 248))
        self.caption_max_question_length = int(
            self.caption_cfg.get("max_question_length", self.qa_max_question_length)
        )

    def __len__(self):
        return len(self.dataset)

    def _build_view_indices(self, num_views):
        if num_views == self.num_views:
            return torch.arange(num_views)
        if num_views > self.num_views:
            return torch.randperm(num_views)[: self.num_views]
        pad_indices = torch.randint(0, num_views, (self.num_views - num_views,))
        return torch.cat([torch.arange(num_views), pad_indices], dim=0)

    def _encode_target(self, target_text, max_target_length):
        encoded = self.tokenizer(
            target_text,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)
        decoder_input_ids = input_ids[:-1].contiguous()
        labels = input_ids[1:].contiguous()
        labels = labels.masked_fill(attention_mask[1:] == 0, -100)
        return decoder_input_ids, labels

    def _encode_question(self, sentence, max_question_length):
        encoded = self.tokenizer(
            sentence,
            max_length=max_question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "question_input_ids": encoded.input_ids.squeeze(0),
            "question_attention_mask": encoded.attention_mask.squeeze(0),
        }

    def _process_images(self, images):
        return self.image_processor(
            images,
            do_center_crop=False,
            do_resize=True,
            size={"height": 224, "width": 224},
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

    def __getitem__(self, idx):
        base = self.dataset[idx]
        out = {}
        for key, value in base.items():
            if torch.is_tensor(value):
                out[key] = value.contiguous().clone()
            else:
                out[key] = value

        is_caption_sample = "caption_text" in out and "sentence" not in out
        max_target_length = (
            self.caption_max_target_length if is_caption_sample else self.qa_max_target_length
        )
        max_question_length = (
            self.caption_max_question_length if is_caption_sample else self.qa_max_question_length
        )

        if torch.is_tensor(out.get("point_map")):
            view_indices = self._build_view_indices(out["point_map"].shape[0])
            out["point_map"] = out["point_map"][view_indices].contiguous().clone()
            if torch.is_tensor(out.get("view_ids")):
                out["view_ids"] = out["view_ids"][view_indices].contiguous().clone()
            if torch.is_tensor(out.get("images")):
                out["images"] = self._process_images(out["images"][view_indices])
            if isinstance(out.get("file_names"), list):
                out["file_names"] = [out["file_names"][int(i)] for i in view_indices.tolist()]

        target_text = out.get("target_text", "")
        decoder_input_ids, decoder_labels = self._encode_target(target_text, max_target_length)
        out["decoder_input_ids"] = decoder_input_ids
        out["decoder_labels"] = decoder_labels
        out["target_text"] = target_text

        if "sentence" in out:
            out.update(self._encode_question(out["sentence"], max_question_length))

        return out

    def collate_fn(self, batch_list):
        collated = {}
        keys = batch_list[0].keys()
        for key in keys:
            values = [sample[key] for sample in batch_list]
            if torch.is_tensor(values[0]):
                collated[key] = torch.stack([value.contiguous().clone() for value in values], dim=0)
            else:
                collated[key] = values
        return collated

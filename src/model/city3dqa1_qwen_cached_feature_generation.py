"""Qwen QA/caption generation model using cached FG-CLIP features."""

from pathlib import Path

import torch

from model.build import BaseModel, MODEL_REGISTRY
from modules.build import build_module
from optim.utils import no_decay_param_group
from transformers import AutoTokenizer


@MODEL_REGISTRY.register()
class City3DQA1QwenCachedFeatureGeneration(BaseModel):
    """Build only the Qwen decoder; FG-CLIP is never instantiated here."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        model_path = cfg.model.get("qwen_model_path")
        if not model_path:
            model_path = str(Path(__file__).resolve().parents[2] / "pretrained" / "Qwen3.8-2B-Distill")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.head_list = cfg.model.heads.head_list
        for head_name in self.head_list:
            setattr(self, head_name, build_module("heads", getattr(cfg.model.heads, head_name)))
        self._decoder_head_name = self.head_list[0]

    def set_downstream_mode(self):
        if self.cfg.model.get("train_pm_encoder", False):
            raise ValueError(
                "Cached FG-CLIP features are fixed; model.train_pm_encoder must be False."
            )
        for head_name in self.head_list:
            getattr(self, head_name).train()

    def forward(self, data_dict, mode=None):
        scene_tokens = data_dict["scene_features"]
        data_dict["inter_view_pm_embed"] = scene_tokens
        data_dict["scene_pm_embed"] = scene_tokens.mean(dim=1)
        decoder_head = getattr(self, self._decoder_head_name)
        data_dict["text_logits"] = decoder_head(
            scene_tokens=scene_tokens,
            input_ids=data_dict["decoder_input_ids"],
            attention_mask=data_dict.get("decoder_attention_mask"),
        )
        if not self.training:
            generated_ids = self._generate(
                decoder_head,
                scene_tokens,
                data_dict,
                int(self.cfg.model.get("generation", {}).get("max_new_tokens", 256)),
            )
            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            if "sentence" in data_dict:
                data_dict["answer_pred"] = decoded
            else:
                data_dict["caption_pred"] = decoded
        return data_dict

    @staticmethod
    def _generate(head, scene_tokens, data_dict, max_new_tokens):
        input_ids = data_dict.get("generation_input_ids", data_dict["decoder_input_ids"])
        attention_mask = data_dict.get("generation_attention_mask")
        eos_token_id = head.language_model.config.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0]
        outputs = []
        for row in range(input_ids.shape[0]):
            if attention_mask is None:
                prompt = input_ids[row : row + 1]
            else:
                valid_length = int(attention_mask[row].sum().item())
                prompt = input_ids[row : row + 1, :valid_length]
            generated = prompt
            generated_mask = torch.ones_like(prompt)
            scene = scene_tokens[row : row + 1]
            for _ in range(max_new_tokens):
                logits = head(scene, generated, generated_mask)
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                generated_mask = torch.cat(
                    [generated_mask, torch.ones_like(next_token)], dim=1
                )
                if int(next_token.item()) == int(eos_token_id):
                    break
            outputs.append(generated[:, prompt.shape[1] :].squeeze(0))

        pad_id = int(head.language_model.config.pad_token_id)
        return torch.nn.utils.rnn.pad_sequence(
            outputs, batch_first=True, padding_value=pad_id
        )

    def get_opt_params(self):
        grouped_parameters = []
        for head_name in self.head_list:
            head_cfg = getattr(self.cfg.model.heads, head_name)
            lr = head_cfg.get("lr") or self.cfg.solver.lr
            grouped_parameters += no_decay_param_group(
                getattr(self, head_name).named_parameters(), lr
            )
        return grouped_parameters

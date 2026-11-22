"""Frozen Qwen language model with LoRA and a scene-token input adapter."""

import torch
from torch import nn

from modules.build import HEADS_REGISTRY


@HEADS_REGISTRY.register()
class City3DQA1QwenLoRATextHead(nn.Module):
    def __init__(
        self,
        model_path,
        scene_dim=512,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=None,
        gradient_checkpointing=True,
    ):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        self.language_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        )
        for parameter in self.language_model.parameters():
            parameter.requires_grad = False

        if gradient_checkpointing and hasattr(self.language_model, "gradient_checkpointing_enable"):
            self.language_model.gradient_checkpointing_enable()
            self.language_model.enable_input_require_grads()

        lora_config = LoraConfig(
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ],
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        text_config = getattr(self.language_model.config, "text_config", self.language_model.config)
        hidden_size = int(text_config.hidden_size)
        self.scene_projection = nn.Sequential(
            nn.Linear(int(scene_dim), hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, scene_tokens, input_ids, attention_mask=None):
        language_model = self.language_model
        embeddings = language_model.get_input_embeddings()(input_ids)
        scene_prefix = self.scene_projection(scene_tokens.to(self.scene_projection[0].weight.dtype))
        scene_prefix = scene_prefix.to(embeddings.dtype)
        inputs_embeds = torch.cat([scene_prefix, embeddings], dim=1)
        batch_size, scene_len = scene_tokens.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prefix_mask = torch.ones(
            batch_size, scene_len, dtype=attention_mask.dtype, device=attention_mask.device
        )
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        output = language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            use_cache=False,
            return_dict=True,
        )
        return output.logits[:, scene_len:, :]

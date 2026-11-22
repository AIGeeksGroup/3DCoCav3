"""Qwen LoRA head with an explicit text-to-scene cross-attention block.

The Qwen language model receives only text token ids.  Cached FG-CLIP scene
tokens are projected to the Qwen hidden size and are read by the text hidden
states through a separate visual cross-attention module.  This keeps visual
tokens out of the causal language-model prefix sequence.
"""

import torch
from torch import nn

from modules.build import HEADS_REGISTRY
from .city3dqa1_qwen_lora_text_head import City3DQA1QwenLoRATextHead


@HEADS_REGISTRY.register()
class City3DQA1QwenVisualCrossAttentionHead(City3DQA1QwenLoRATextHead):
    """Frozen Qwen + LoRA, scene adapter, and explicit visual cross-attention."""

    def __init__(
        self,
        model_path,
        scene_dim=512,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=None,
        gradient_checkpointing=True,
        cross_attention_heads=8,
        cross_attention_dropout=0.0,
        cross_attention_gate_init=0.1,
    ):
        super().__init__(
            model_path=model_path,
            scene_dim=scene_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            gradient_checkpointing=gradient_checkpointing,
        )

        text_config = getattr(self.language_model.config, "text_config", self.language_model.config)
        hidden_size = int(text_config.hidden_size)
        cross_attention_heads = int(cross_attention_heads)
        if cross_attention_heads <= 0 or hidden_size % cross_attention_heads != 0:
            raise ValueError(
                "cross_attention_heads must be positive and divide the Qwen hidden size "
                f"({hidden_size}); got {cross_attention_heads}."
            )

        # These are the only newly introduced multimodal parameters.  The
        # Qwen base remains frozen while its LoRA parameters stay trainable.
        self.visual_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=cross_attention_heads,
            dropout=float(cross_attention_dropout),
            batch_first=True,
        )
        self.cross_attention_norm = nn.LayerNorm(hidden_size)
        self.cross_attention_gate = nn.Parameter(
            torch.tensor(float(cross_attention_gate_init), dtype=torch.float32)
        )

    def forward(
        self,
        scene_tokens,
        input_ids,
        attention_mask=None,
        scene_attention_mask=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Run the LoRA-instrumented Qwen backbone on text only.  Calling the
        # backbone directly returns only its final hidden state instead of
        # retaining every transformer layer through output_hidden_states=True.
        base_language_model = self.language_model.get_base_model()
        text_backbone = getattr(base_language_model, "model", None)
        if text_backbone is None:
            # This fallback keeps the head usable with PEFT/model variants
            # that do not expose a separate backbone attribute.
            language_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = getattr(language_output, "hidden_states", None)
            text_hidden = hidden_states[-1] if hidden_states else getattr(
                language_output, "last_hidden_state", None
            )
        else:
            language_output = text_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            text_hidden = getattr(language_output, "last_hidden_state", None)
        if text_hidden is None:
            raise RuntimeError("Qwen output did not contain final text hidden states.")

        # MHA is evaluated in fp32 for stable newly initialized adapter and
        # attention weights, then returned to the Qwen activation dtype.
        text_query = text_hidden.float()
        scene_memory = self.scene_projection(scene_tokens.float()).float()
        key_padding_mask = None
        if scene_attention_mask is not None:
            key_padding_mask = scene_attention_mask.to(scene_memory.device).eq(0)

        attended_scene, _ = self.visual_cross_attention(
            query=text_query,
            key=scene_memory,
            value=scene_memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended_scene = self.cross_attention_norm(attended_scene)
        fused_hidden = text_query + self.cross_attention_gate * attended_scene

        output_embeddings = self.language_model.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("Qwen model does not expose an output embedding/LM head.")
        return output_embeddings(fused_hidden.to(text_hidden.dtype))

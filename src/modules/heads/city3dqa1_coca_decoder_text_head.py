"""CoCa-style text decoder head for City3DQA1 caption and QA generation."""

import torch
from torch import nn

from modules.build import HEADS_REGISTRY


@HEADS_REGISTRY.register()
class City3DQA1CoCaDecoderTextHead(nn.Module):
    """Generate text with causal self-attention and cross-attention to scene tokens."""

    def __init__(
        self,
        vocab_size=49408,
        d_model=512,
        scene_dim=512,
        question_dim=1024,
        num_layers=4,
        num_heads=8,
        ff_mult=4,
        dropout=0.1,
        max_text_len=96,
        max_question_len=256,
        max_view_id=1024,
        pad_token_id=49407,
        bos_token_id=49406,
        eos_token_id=49407,
        use_question_tokens=True,
        use_view_position=True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_text_len = max_text_len
        self.max_question_len = max_question_len
        self.max_view_id = max_view_id
        self.use_question_tokens = use_question_tokens
        self.use_view_position = use_view_position

        self.scene_projection = nn.Linear(scene_dim, d_model)
        self.question_projection = nn.Linear(question_dim, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.text_position_embedding = nn.Embedding(max_text_len, d_model)
        self.question_position_embedding = nn.Embedding(max_question_len, d_model)
        self.view_position_embedding = nn.Embedding(max_view_id, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.memory_norm = nn.LayerNorm(d_model)
        self.text_norm = nn.LayerNorm(d_model)
        self.to_logits = nn.Linear(d_model, vocab_size, bias=False)
        self.to_logits.weight = self.token_embedding.weight

    def _causal_mask(self, seq_len, device):
        return torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(1)

    def _embed_question_ids(self, question_input_ids):
        seq_len = question_input_ids.shape[1]
        if seq_len > self.max_question_len:
            raise ValueError(f"question_input_ids length {seq_len} exceeds max_question_len={self.max_question_len}.")

        positions = torch.arange(seq_len, device=question_input_ids.device)
        positions = positions.unsqueeze(0).expand(question_input_ids.shape[0], seq_len)
        return self.token_embedding(question_input_ids) + self.question_position_embedding(positions)

    def _build_memory(
        self,
        scene_tokens,
        question_input_ids=None,
        question_attention_mask=None,
        question_tokens=None,
        question_padding_mask=None,
        scene_attention_mask=None,
        scene_padding_mask=None,
        view_ids=None,
    ):
        dtype = self.scene_projection.weight.dtype
        scene_memory = self.scene_projection(scene_tokens.to(dtype))

        if self.use_view_position and view_ids is not None:
            view_ids = view_ids.to(scene_memory.device).long().clamp(0, self.max_view_id - 1)
            scene_memory = scene_memory + self.view_position_embedding(view_ids)

        batch_size, num_views, _ = scene_memory.shape
        if scene_padding_mask is None:
            if scene_attention_mask is None:
                scene_padding_mask = torch.zeros(
                    batch_size,
                    num_views,
                    dtype=torch.bool,
                    device=scene_memory.device,
                )
            else:
                scene_padding_mask = scene_attention_mask.to(scene_memory.device).ne(1)
        else:
            scene_padding_mask = scene_padding_mask.to(scene_memory.device).bool()

        memory_parts = []
        mask_parts = []
        if self.use_question_tokens and question_input_ids is not None:
            question_memory = self._embed_question_ids(question_input_ids).to(dtype)
            memory_parts.append(question_memory)
            if question_attention_mask is None:
                question_padding_mask = torch.zeros(
                    question_memory.shape[:2],
                    dtype=torch.bool,
                    device=question_memory.device,
                )
            else:
                question_padding_mask = question_attention_mask.to(question_memory.device).ne(1)
            mask_parts.append(question_padding_mask)
        elif self.use_question_tokens and question_tokens is not None:
            question_memory = self.question_projection(question_tokens.to(dtype))
            memory_parts.append(question_memory)
            if question_padding_mask is None:
                question_padding_mask = torch.zeros(
                    question_memory.shape[:2],
                    dtype=torch.bool,
                    device=question_memory.device,
                )
            mask_parts.append(question_padding_mask.to(question_memory.device))

        memory_parts.append(scene_memory)
        mask_parts.append(scene_padding_mask)

        memory = self.memory_norm(torch.cat(memory_parts, dim=1))
        memory_padding_mask = torch.cat(mask_parts, dim=1)
        return memory, memory_padding_mask

    def forward(
        self,
        scene_tokens,
        decoder_input_ids,
        question_input_ids=None,
        question_attention_mask=None,
        question_tokens=None,
        question_padding_mask=None,
        scene_attention_mask=None,
        scene_padding_mask=None,
        view_ids=None,
    ):
        memory, memory_padding_mask = self._build_memory(
            scene_tokens,
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask,
            question_tokens=question_tokens,
            question_padding_mask=question_padding_mask,
            scene_attention_mask=scene_attention_mask,
            scene_padding_mask=scene_padding_mask,
            view_ids=view_ids,
        )

        seq_len = decoder_input_ids.shape[1]
        if seq_len > self.max_text_len:
            raise ValueError(f"decoder_input_ids length {seq_len} exceeds max_text_len={self.max_text_len}.")

        positions = torch.arange(seq_len, device=decoder_input_ids.device)
        positions = positions.unsqueeze(0).expand(decoder_input_ids.shape[0], seq_len)
        text_tokens = self.token_embedding(decoder_input_ids) + self.text_position_embedding(positions)
        text_tokens = self.text_norm(text_tokens)

        decoded = self.decoder(
            tgt=text_tokens,
            memory=memory,
            tgt_mask=self._causal_mask(seq_len, decoder_input_ids.device),
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.to_logits(decoded)

    @torch.no_grad()
    def generate(
        self,
        scene_tokens,
        question_input_ids=None,
        question_attention_mask=None,
        question_tokens=None,
        question_padding_mask=None,
        scene_attention_mask=None,
        scene_padding_mask=None,
        view_ids=None,
        max_new_tokens=32,
    ):
        batch_size = scene_tokens.shape[0]
        device = scene_tokens.device
        generated = torch.full(
            (batch_size, 1),
            int(self.bos_token_id),
            dtype=torch.long,
            device=device,
        )

        for _ in range(max_new_tokens):
            logits = self.forward(
                scene_tokens=scene_tokens,
                decoder_input_ids=generated,
                question_input_ids=question_input_ids,
                question_attention_mask=question_attention_mask,
                question_tokens=question_tokens,
                question_padding_mask=question_padding_mask,
                scene_attention_mask=scene_attention_mask,
                scene_padding_mask=scene_padding_mask,
                view_ids=view_ids,
            )
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token.squeeze(1) == int(self.eos_token_id)):
                break

        return generated

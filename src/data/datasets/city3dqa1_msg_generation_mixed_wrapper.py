"""Batch QA and MSG/pseudo-caption targets without mixing their conditioning."""

import torch
from torch.nn.utils.rnn import pad_sequence

from .city3dqa1_coca_decoder_pt_cocaloss_wrapper import (
    City3DQA1QAAndCaptionUrbanCoCaDecoderPTWrapper,
)
from .dataset_wrapper import DATASETWRAPPER_REGISTRY


@DATASETWRAPPER_REGISTRY.register()
class City3DQA1MSGGenerationMixedWrapper(
    City3DQA1QAAndCaptionUrbanCoCaDecoderPTWrapper
):
    """Keep per-task truncation limits, then right-pad only for batching.

    Caption rows have no question: their question memory is fully masked in a
    mixed batch. Pure caption evaluation retains the absence of ``sentence``
    so the existing model emits ``caption_pred``, not ``answer_pred``.
    """

    def collate_fn(self, batch_list):
        if not batch_list:
            raise ValueError("Cannot collate an empty QA/caption batch.")

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("QA/caption batching requires a tokenizer pad token.")
        sequence_padding = {
            "decoder_input_ids": pad_id,
            "decoder_labels": -100,
            "question_input_ids": pad_id,
            "question_attention_mask": 0,
        }
        optional_questions = {"question_input_ids", "question_attention_mask"}
        collated = {}
        keys = dict.fromkeys(key for sample in batch_list for key in sample)
        for key in keys:
            values = [sample.get(key) for sample in batch_list]
            exemplar = next(value for value in values if value is not None)
            if key in sequence_padding:
                if key not in optional_questions and any(v is None for v in values):
                    raise ValueError(f"Every training sample must provide {key}.")
                # Missing caption questions contribute padding only; no target
                # caption is ever copied into question/encoder inputs.
                values = [exemplar.new_empty(0) if v is None else v for v in values]
                collated[key] = pad_sequence(
                    values, batch_first=True, padding_value=sequence_padding[key]
                )
            elif torch.is_tensor(exemplar):
                if key == "question_type":
                    values = [
                        torch.full_like(exemplar, -1) if v is None else v
                        for v in values
                    ]
                elif any(v is None for v in values):
                    raise ValueError(f"Missing required tensor {key} in QA/caption batch.")
                collated[key] = torch.stack(values, dim=0)
            else:
                # Preserve row order and task-specific metadata; a field absent
                # from the entire batch stays absent (especially ``sentence``).
                collated[key] = ["" if v is None else v for v in values]
        return collated

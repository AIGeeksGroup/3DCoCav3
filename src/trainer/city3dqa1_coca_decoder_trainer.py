"""Trainer for City3DQA1 CoCa decoder caption and QA generation."""

from trainer.build import TRAINER_REGISTRY
from trainer.default_trainer import DefaultTrainer


@TRAINER_REGISTRY.register()
class City3DQA1CoCaDecoderTrainer(DefaultTrainer):
    """Default downstream trainer with generative text metrics gathered."""

    def _gather_for_metrics(self, data_dict):
        out = {}
        metric_keys = [
            "answer_pred",
            "caption_pred",
            "target_text",
            "caption_text",
            "sentence",
            "scan_id",
            "data_idx",
            "answers",
            "question_type",
        ]
        for key in metric_keys:
            if key in data_dict:
                out[key] = self.accelerator.gather_for_metrics(data_dict[key])
        return out


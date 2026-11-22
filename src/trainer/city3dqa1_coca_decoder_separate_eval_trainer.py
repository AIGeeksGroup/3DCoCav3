"""Trainer with independent QA and scene-caption test evaluation."""

import torch
from tqdm import tqdm

from evaluator.common.build import get_eval
from trainer.build import TRAINER_REGISTRY
from trainer.city3dqa1_coca_decoder_frozen_pm_eval_trainer import (
    City3DQA1CoCaDecoderFrozenPMEvalTrainer,
)


@TRAINER_REGISTRY.register()
class City3DQA1CoCaDecoderSeparateQACaptionEvalTrainer(
    City3DQA1CoCaDecoderFrozenPMEvalTrainer
):
    """Keep QA and caption test sets and metrics completely separate."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.caption_evaluator = get_eval(
            "City3DQA1CaptionUrbanCoCaDecoderPTEval",
            cfg,
            self.accelerator,
        )

    def _test_loaders(self):
        loaders = self.data_loaders["test"]
        if isinstance(loaders, (list, tuple)):
            return list(loaders)
        return [loaders]

    @torch.no_grad()
    def _evaluate_one(self, loader, evaluator, log_mode, result_split):
        self.model.eval()
        pbar = tqdm(
            range(len(loader)),
            disable=(not self.accelerator.is_main_process),
            desc=f"[{log_mode}]",
        )
        for _, data_dict in enumerate(loader):
            data_dict = self.forward(data_dict, mode="qa")
            gathered = self._gather_for_metrics(data_dict)
            if self.accelerator.is_main_process:
                evaluator.update(gathered)
            pbar.update(1)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            _, results = evaluator.record(split=result_split)
            self.log(results, mode=log_mode)
            evaluator.reset()
        else:
            results = None
        return results

    @torch.no_grad()
    def eval_step(self, epoch):
        loaders = self._test_loaders()
        if len(loaders) < 2:
            return super().eval_step(epoch)
        self._evaluate_one(loaders[0], self.evaluator, "test_qa", "test_qa")
        self._evaluate_one(
            loaders[1], self.caption_evaluator, "test_caption", "test_caption"
        )
        return False

    @torch.no_grad()
    def test_step(self):
        loaders = self._test_loaders()
        if len(loaders) < 2:
            return super().test_step()
        qa_results = self._evaluate_one(
            loaders[0], self.evaluator, "test_qa", "test_qa"
        )
        self._evaluate_one(
            loaders[1], self.caption_evaluator, "test_caption", "test_caption"
        )
        return qa_results if self.accelerator.is_main_process else None

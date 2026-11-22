"""Evaluators for City3DQA1 CoCa decoder caption and QA generation."""

import json
import re
from pathlib import Path

from evaluator.common.build import BaseEvaluator, EVALUATOR_REGISTRY


def _normalize_city3dqa1_coca_decoder_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9,\s\-:]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _token_f1_city3dqa1_coca_decoder(pred, gt):
    pred_tokens = _normalize_city3dqa1_coca_decoder_text(pred).split()
    gt_tokens = _normalize_city3dqa1_coca_decoder_text(gt).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = 0
    remaining = list(gt_tokens)
    for token in pred_tokens:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


class _BaseCity3DQA1CoCaDecoderEval(BaseEvaluator):
    def record(self, split=None):
        for key, metrics in self.eval_dict.items():
            if not isinstance(metrics, list):
                continue
            total_value = sum(item[0] for item in metrics)
            total_count = sum(item[1] for item in metrics)
            self.eval_dict[key] = total_value / max(total_count, 1)
            print(key, total_value, total_count)

        if self.save and self.accelerator.is_main_process:
            output_name = "results.json" if split is None else f"{split}_results.json"
            with (self.save_dir / output_name).open("w", encoding="utf-8") as f:
                json.dump(self.eval_results, f, ensure_ascii=False, indent=2)

        self.eval_dict["target_metric"] = self.eval_dict[self.target_metric]
        if self.eval_dict["target_metric"] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict["target_metric"]
        else:
            is_best = False
        self.eval_dict["best_result"] = self.best_result
        return is_best, self.eval_dict


@EVALUATOR_REGISTRY.register()
class City3DQA1QAUrbanCoCaDecoderPTEval(_BaseCity3DQA1CoCaDecoderEval):
    """Normalized exact-match evaluator for City3DQA1 generative QA."""

    def __init__(self, cfg, accelerator, **kwargs):
        self.target_metric = "ans1_acc"
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / self.__class__.__name__
        super().__init__(cfg, accelerator, **kwargs)

    def batch_metrics(self, data_dict, include_count=False):
        preds = data_dict["answer_pred"]
        answer_groups = [answers.split("[answer_seq]") if isinstance(answers, str) else answers for answers in data_dict["answers"]]
        total_count = len(preds)

        correct = 0
        token_f1_sum = 0.0
        single_correct = 0
        single_total = 0
        multi_correct = 0
        multi_total = 0

        for idx, (pred, answers) in enumerate(zip(preds, answer_groups)):
            pred_norm = _normalize_city3dqa1_coca_decoder_text(pred)
            answers_norm = [_normalize_city3dqa1_coca_decoder_text(answer) for answer in answers]
            is_correct = pred_norm in answers_norm
            correct += int(is_correct)
            token_f1_sum += max((_token_f1_city3dqa1_coca_decoder(pred, answer) for answer in answers), default=0.0)

            if "question_type" in data_dict:
                qtype = int(data_dict["question_type"][idx].item())
                if qtype == 0:
                    single_total += 1
                    single_correct += int(is_correct)
                else:
                    multi_total += 1
                    multi_correct += int(is_correct)

        metrics = {
            "ans1_acc": (correct, total_count),
            "answer_token_f1": (token_f1_sum, total_count),
        }
        if "question_type" in data_dict:
            metrics["single_hop_ans1_acc"] = (single_correct, single_total)
            metrics["multi_hop_ans1_acc"] = (multi_correct, multi_total)

        if self.save:
            for idx, pred in enumerate(preds):
                self.eval_results.append(
                    {
                        "scene_id": data_dict["scan_id"][idx],
                        "question_id": data_dict["data_idx"][idx],
                        "question": data_dict["sentence"][idx],
                        "prediction": pred,
                        "answers": answer_groups[idx],
                    }
                )

        if not include_count:
            return {key: value[0] / max(value[1], 1) for key, value in metrics.items()}
        return metrics


@EVALUATOR_REGISTRY.register()
class City3DQA1CaptionUrbanCoCaDecoderPTEval(_BaseCity3DQA1CoCaDecoderEval):
    """Lightweight caption evaluator for City3DQA1 scene-caption generation."""

    def __init__(self, cfg, accelerator, **kwargs):
        self.target_metric = "caption_token_f1"
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / self.__class__.__name__
        super().__init__(cfg, accelerator, **kwargs)

    def batch_metrics(self, data_dict, include_count=False):
        preds = data_dict["caption_pred"]
        targets = data_dict["target_text"]
        total_count = len(preds)
        exact = 0
        token_f1_sum = 0.0
        for pred, target in zip(preds, targets):
            exact += int(
                _normalize_city3dqa1_coca_decoder_text(pred)
                == _normalize_city3dqa1_coca_decoder_text(target)
            )
            token_f1_sum += _token_f1_city3dqa1_coca_decoder(pred, target)

        metrics = {
            "caption_exact_match": (exact, total_count),
            "caption_token_f1": (token_f1_sum, total_count),
        }

        if self.save:
            for idx, pred in enumerate(preds):
                self.eval_results.append(
                    {
                        "scene_id": data_dict["scan_id"][idx],
                        "prediction": pred,
                        "target": targets[idx],
                    }
                )

        if not include_count:
            return {key: value[0] / max(value[1], 1) for key, value in metrics.items()}
        return metrics


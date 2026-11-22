"""50-view QA evaluator for City3DQA1 CoCa decoder generation."""

from collections import OrderedDict
import json
from pathlib import Path

from evaluator.common.build import EVALUATOR_REGISTRY
from evaluator.vqa3d.city3dqa1_coca_decoder_eval import (
    City3DQA1QAUrbanCoCaDecoderPTEval,
    _normalize_city3dqa1_coca_decoder_text,
    _token_f1_city3dqa1_coca_decoder,
)


@EVALUATOR_REGISTRY.register()
class City3DQA1QAUrban50ViewCoCaDecoderPTEval(City3DQA1QAUrbanCoCaDecoderPTEval):
    """Record generated-QA accuracy plus explicit correct/total counts."""

    def __init__(self, cfg, accelerator, **kwargs):
        super().__init__(cfg, accelerator, **kwargs)
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / self.__class__.__name__

    def _as_list(self, value):
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        return list(value)

    def batch_metrics(self, data_dict, include_count=False):
        preds = self._as_list(data_dict["answer_pred"])
        answers_raw = self._as_list(data_dict["answers"])
        scan_ids = self._as_list(data_dict["scan_id"])
        data_idxs = self._as_list(data_dict["data_idx"])
        questions = self._as_list(data_dict.get("sentence", [""] * len(preds)))

        common_len = min(len(preds), len(answers_raw), len(scan_ids), len(data_idxs), len(questions))
        records = OrderedDict()
        for idx in range(common_len):
            key = (str(scan_ids[idx]), str(data_idxs[idx]))
            if key in records:
                continue

            answers = (
                answers_raw[idx].split("[answer_seq]")
                if isinstance(answers_raw[idx], str)
                else list(answers_raw[idx])
            )
            records[key] = {
                "pred": preds[idx],
                "answers": answers,
                "scan_id": scan_ids[idx],
                "data_idx": data_idxs[idx],
                "question": questions[idx],
                "question_type": None,
            }

        if "question_type" in data_dict:
            question_types = self._as_list(data_dict["question_type"])
            if len(question_types) == len(records):
                for record, qtype in zip(records.values(), question_types):
                    record["question_type"] = int(qtype)
            else:
                print(
                    "[City3DQA1QAUrban50ViewCoCaDecoderPTEval] "
                    f"Skip hop metrics for gathered batch because question_type={len(question_types)} "
                    f"but unique QA records={len(records)}."
                )

        correct = 0
        token_f1_sum = 0.0
        single_correct = 0
        single_total = 0
        multi_correct = 0
        multi_total = 0

        for record in records.values():
            pred_norm = _normalize_city3dqa1_coca_decoder_text(record["pred"])
            answers_norm = [_normalize_city3dqa1_coca_decoder_text(answer) for answer in record["answers"]]
            is_correct = pred_norm in answers_norm
            correct += int(is_correct)
            token_f1_sum += max(
                (_token_f1_city3dqa1_coca_decoder(record["pred"], answer) for answer in record["answers"]),
                default=0.0,
            )

            qtype = record["question_type"]
            if qtype is not None:
                if qtype == 0:
                    single_total += 1
                    single_correct += int(is_correct)
                else:
                    multi_total += 1
                    multi_correct += int(is_correct)

        total_count = len(records)
        metrics = {
            "ans1_acc": (correct, total_count),
            "answer_token_f1": (token_f1_sum, total_count),
        }
        if single_total + multi_total > 0:
            metrics["single_hop_ans1_acc"] = (single_correct, single_total)
            metrics["multi_hop_ans1_acc"] = (multi_correct, multi_total)

        if self.save:
            for record in records.values():
                self.eval_results.append(
                    {
                        "scene_id": record["scan_id"],
                        "question_id": record["data_idx"],
                        "question": record["question"],
                        "prediction": record["pred"],
                        "answers": record["answers"],
                    }
                )

        if not include_count:
            return {key: value[0] / max(value[1], 1) for key, value in metrics.items()}
        return metrics

    def record(self, split=None):
        count_summary = {}
        for key, metrics in self.eval_dict.items():
            if not isinstance(metrics, list):
                continue
            total_value = sum(item[0] for item in metrics)
            total_count = sum(item[1] for item in metrics)
            self.eval_dict[key] = total_value / max(total_count, 1)
            print(key, total_value, total_count)

            if key == "ans1_acc":
                count_summary["ans1_correct_count"] = total_value
                count_summary["ans1_total_count"] = total_count
                count_summary["ans1_acc_correct_count"] = total_value
                count_summary["ans1_acc_total_count"] = total_count
            elif key == "answer_token_f1":
                count_summary["answer_token_f1_sum"] = total_value
                count_summary["answer_token_f1_total_count"] = total_count
            elif key == "single_hop_ans1_acc":
                count_summary["single_hop_ans1_correct_count"] = total_value
                count_summary["single_hop_ans1_total_count"] = total_count
                count_summary["single_hop_ans1_acc_correct_count"] = total_value
                count_summary["single_hop_ans1_acc_total_count"] = total_count
            elif key == "multi_hop_ans1_acc":
                count_summary["multi_hop_ans1_correct_count"] = total_value
                count_summary["multi_hop_ans1_total_count"] = total_count
                count_summary["multi_hop_ans1_acc_correct_count"] = total_value
                count_summary["multi_hop_ans1_acc_total_count"] = total_count

        self.eval_dict.update(count_summary)

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

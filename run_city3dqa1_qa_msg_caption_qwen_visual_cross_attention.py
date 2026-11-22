"""Qwen QA entry point with retrieved captions and visual cross-attention."""

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import data.datasets.city3dqa1_msg_caption_qa  # noqa: F401,E402
import data.datasets.city3dqa1_qwen_qa_retrieved_caption_wrapper  # noqa: F401,E402
import evaluator.vqa3d.city3dqa1_coca_decoder_eval  # noqa: F401,E402
import evaluator.vqa3d.city3dqa1_qa_urban_50view_coca_decoder_eval  # noqa: F401,E402
import model.city3dqa1_qwen_visual_cross_attention_generation  # noqa: F401,E402
import modules.heads.city3dqa1_qwen_visual_cross_attention_head  # noqa: F401,E402
import optim.loss.city3dqa1_coca_decoder_losses  # noqa: F401,E402
import trainer.city3dqa1_coca_decoder_frozen_pm_eval_trainer  # noqa: F401,E402
import trainer.city3dqa1_coca_decoder_separate_eval_trainer  # noqa: F401,E402

import hydra  # noqa: E402
from run_city3dqa1_qa_msg_caption_mixed_generation import main as _original_main  # noqa: E402


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parent / "configs" / "finetune"),
    config_name="city3dqa1_qa_urban_coca_decoder_pt",
)
def main(cfg):
    return _original_main.__wrapped__(cfg)


if __name__ == "__main__":
    main()

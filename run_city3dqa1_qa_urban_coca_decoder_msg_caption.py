"""City3DQA1 QA + MSG scene-caption training entry point."""

from datetime import datetime
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import data.datasets.city3dqa1_msg_caption_qa  # noqa: F401
import data.datasets.city3dqa1_coca_decoder_pt_cocaloss_wrapper  # noqa: F401
import evaluator.vqa3d.city3dqa1_coca_decoder_eval  # noqa: F401
import evaluator.vqa3d.city3dqa1_qa_urban_50view_coca_decoder_eval  # noqa: F401
import model.city3dqa1_msg_caption_generation  # noqa: F401
import modules.heads.city3dqa1_coca_decoder_text_head  # noqa: F401
import optim.loss.city3dqa1_coca_decoder_losses  # noqa: F401
import trainer.city3dqa1_coca_decoder_frozen_pm_eval_trainer  # noqa: F401
import trainer.city3dqa1_coca_decoder_separate_eval_trainer  # noqa: F401

import hydra
from omegaconf import OmegaConf, open_dict
import wandb

from common.misc import make_dir, rgetattr
from trainer.build import build_trainer


@hydra.main(
    version_base=None,
    config_path="./configs/finetune",
    config_name="city3dqa1_qa_urban_coca_decoder_pt",
)
def main(cfg):
    if cfg.resume:
        assert Path(cfg.exp_dir).exists(), f"Resuming failed: {cfg.exp_dir} does not exist."
        cfg = OmegaConf.load(Path(cfg.exp_dir) / "config.yaml")
        cfg.resume = True
    else:
        run_id = wandb.util.generate_id()
        with open_dict(cfg):
            cfg.logger.run_id = run_id

    OmegaConf.resolve(cfg)
    naming_keys = [cfg.name]
    for name in cfg.get("naming_keywords", []):
        if name == "time":
            continue
        if name == "task":
            naming_keys.append(cfg.task)
            naming_keys.append("+".join(str(item) for item in rgetattr(cfg, "data.train")))
        elif name == "dataloader.batchsize":
            naming_keys.append(f"b{rgetattr(cfg, name) * rgetattr(cfg, 'num_gpu')}")
        else:
            value = str(rgetattr(cfg, name))
            if value:
                naming_keys.append(value)

    exp_name = "_".join(naming_keys)
    if rgetattr(cfg, "debug.flag", False):
        exp_name = "Debug_City3DQA1_MSG_Caption"
    print(exp_name)

    if not cfg.exp_dir:
        cfg.exp_dir = Path(cfg.base_dir) / exp_name / datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    else:
        cfg.exp_dir = Path(cfg.exp_dir)
    make_dir(cfg.exp_dir)
    OmegaConf.save(cfg, cfg.exp_dir / "config.yaml")
    build_trainer(cfg).run()


if __name__ == "__main__":
    main()

"""City3DQA1 CoCa decoder trainer that keeps the frozen PM encoder in eval mode."""

from tqdm import tqdm

from trainer.build import TRAINER_REGISTRY
from trainer.city3dqa1_coca_decoder_trainer import City3DQA1CoCaDecoderTrainer


@TRAINER_REGISTRY.register()
class City3DQA1CoCaDecoderFrozenPMEvalTrainer(City3DQA1CoCaDecoderTrainer):
    """Train the decoder while forcing a frozen pm_encoder to stay in eval mode."""

    def __init__(self, cfg):
        self.train_pm_encoder = cfg.model.get("train_pm_encoder", False)
        super().__init__(cfg)

    def _keep_frozen_pm_encoder_eval(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        if self.train_pm_encoder:
            return
        if not hasattr(model, "pm_encoder"):
            return

        model.pm_encoder.eval()
        for param in model.pm_encoder.parameters():
            param.requires_grad = False

    def train_step(self, epoch):
        self.model.train()
        self._keep_frozen_pm_encoder_eval()

        loader = self.data_loaders["train"]
        pbar = tqdm(
            range(len(loader)),
            disable=(not self.accelerator.is_main_process),
            desc=f"[Epoch {epoch + 1}/{self.epochs}]",
        )
        for i, data_dict in enumerate(loader):
            with self.accelerator.accumulate(self.model):
                self._keep_frozen_pm_encoder_eval()
                data_dict["cur_step"] = epoch * len(loader) + i
                data_dict["total_steps"] = self.total_steps
                data_dict = self.forward(data_dict, mode="qa")
                loss, losses = self.loss(data_dict)
                self.backward(loss)

                self.global_step += 1
                log_dict = {"step": self.global_step}
                log_dict.update(losses)
                self.log(log_dict, mode="train")
                pbar.update(1)

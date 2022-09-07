# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.metrics import mse


class MAELitModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.005,
        warmup_epochs: int = 5,
        max_epochs: int = 30,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        mask_ratio: float = 0.5,
        reconstruct_all=False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

    def forward(self, x, y, variables, out_variables, ):
        with torch.no_grad():
            pred, mask = self.net.pred(x, y, variables, out_variables, self.hparams.mask_ratio)
        return pred, mask

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def training_step(self, batch: Any, batch_idx: int):
        x, y, variables, out_variables = batch
        # loss can either be mse or lat_weighted_mse
        loss_dict, _, _ = self.net.forward(
            x, y, variables, out_variables, mse, self.lat, self.hparams.mask_ratio, self.hparams.reconstruct_all
        )
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        return loss_dict

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, variables, out_variables = batch
        # loss can either be mse or lat_weighted_mse
        loss_dict, _, _ = self.net.forward(
            x, y, variables, out_variables, mse, self.lat, self.hparams.mask_ratio, self.hparams.reconstruct_all
        )
        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    # def validation_epoch_end(self, outputs: List[Any]):
    #     acc = self.val_acc.compute()  # get val accuracy from current epoch
    #     self.val_acc_best.update(acc)
    #     self.log(
    #         "val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True
    #     )

    #     self.val_acc.reset()  # reset val accuracy for next epoch

    def test_step(self, batch: Any, batch_idx: int):
        x, y, variables, out_variables = batch
        # loss can either be mse or lat_weighted_mse
        loss_dict, _, _ = self.net.forward(
            x, y, variables, out_variables, mse, self.lat, self.hparams.mask_ratio, self.hparams.reconstruct_all
        )
        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return loss_dict

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "weight_decay": self.hparams.weight_decay,
                },
                {"params": no_decay, "lr": self.hparams.lr, "weight_decay": 0},
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

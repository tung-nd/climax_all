# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any, List

import torch
from pytorch_lightning import LightningModule


class MAELitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.005,
        mask_ratio: float = 0.5,
        reconstruct_all = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

    def forward(self, x):
        with torch.no_grad():
            _, pred, mask = self.net.forward(x, self.hparams.mask_ratio, self.hparams.reconstruct_all)
        mask = mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])
        pred = self.net.unpatchify(pred)
        mask = self.net.unpatchify(mask)
        return pred, mask

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict, _, _ = self.net.forward(batch, self.hparams.mask_ratio, self.hparams.reconstruct_all)
        for var in loss_dict.keys():
            self.log("train/" + var, loss_dict[var], on_step=True, on_epoch=False, prog_bar=True)
        return loss_dict

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict, _, _ = self.net.forward(batch, self.hparams.mask_ratio, self.hparams.reconstruct_all)
        for var in loss_dict.keys():
            self.log("val/" + var, loss_dict[var], on_step=False, on_epoch=True, prog_bar=False)
        return loss_dict

    # def validation_epoch_end(self, outputs: List[Any]):
    #     acc = self.val_acc.compute()  # get val accuracy from current epoch
    #     self.val_acc_best.update(acc)
    #     self.log(
    #         "val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True
    #     )

    #     self.val_acc.reset()  # reset val accuracy for next epoch

    def test_step(self, batch: Any, batch_idx: int):
        loss_dict, _, _ = self.net.forward(batch, self.hparams.mask_ratio, self.hparams.reconstruct_all)
        for var in loss_dict.keys():
            self.log("test/" + var, loss_dict[var], on_step=False, on_epoch=True)
        return loss_dict

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

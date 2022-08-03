# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any

import torch
from pytorch_lightning import LightningModule
from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.metrics import lat_weighted_acc, lat_weighted_rmse, mse


class ViTPrecipLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        precip_net: torch.nn.Module,
        detach_net,
        pretrained_path: str,
        lr: float = 0.001,
        weight_decay: float = 0.005,
        warmup_epochs: int = 5,
        max_epochs: int = 30,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.precip_net = precip_net
        if len(pretrained_path) > 0:
            self.load_mae_weights(pretrained_path)

    def load_mae_weights(self, pretrained_path):
        checkpoint = torch.load(pretrained_path)

        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        state_dict = self.state_dict()
        checkpoint_keys = list(checkpoint_model.keys())
        for k in checkpoint_keys:
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def forward(self, x):
        return self.net.predict(x)

    def training_step(self, batch: Any, batch_idx: int):
        x, y, precip = batch

        # next-step prediction loss
        pred_loss, pred = self.net.forward(x, y, [mse])
        pred_loss = pred_loss[0]

        # precipitation prediction on top of next-step prediction
        if self.hparams.detach_net == True:
            pred = pred.detach()
        precip_loss, _ = self.precip_net.forward(pred, precip, [mse])
        precip_loss = precip_loss[0]

        for var in pred_loss.keys():
            self.log(
                "train/" + var,
                pred_loss[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        self.log(
            "train/" + list(precip_loss.keys())[0],
            precip_loss[list(precip_loss.keys())[0]],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        loss_dict = {}
        loss_dict["loss"] = pred_loss["loss"] + precip_loss["loss"]
        
        return loss_dict

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, precip = batch
        pred_steps = y.shape[1]
        pred_loss, preds = self.net.rollout(x, y, pred_steps, [lat_weighted_rmse, lat_weighted_acc])

        precip_pred = []
        with torch.no_grad():
            for step in range(preds.shape[1]):
                precip_pred.append(self.precip_net.predict(preds[:, step]))
        precip_pred = torch.stack(precip_pred, dim=1)
        precip_pred_loss = [
            lat_weighted_rmse(precip_pred, precip, ['total_precipitation']),
            lat_weighted_acc(precip_pred, precip, ['total_precipitation'])
        ]

        loss_dict = {}
        for d in pred_loss:
            for k in d.keys():
                loss_dict[k] = d[k]
        for d in precip_pred_loss:
            for k in d.keys():
                loss_dict[k] = d[k]

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
        x, y = batch
        pred_steps = y.shape[1]
        all_loss_dicts = self.net.rollout(x, y, pred_steps, [lat_weighted_rmse, lat_weighted_acc])

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

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

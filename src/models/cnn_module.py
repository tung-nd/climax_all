from typing import Any, Union

import torch
from pytorch_lightning import LightningModule
from src.models.components.resnet import ResNet
from src.models.components.unet import Unet
from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.metrics import (lat_weighted_acc, lat_weighted_mse,
                               lat_weighted_mse_val, lat_weighted_rmse)
from torchvision.transforms import transforms


class CNNLitModule(LightningModule):
    def __init__(
        self,
        net: Union[ResNet, Unet],
        pretrained_path: str,
        lr: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        weight_decay: float = 0.005,
        warmup_epochs: int = 5,
        max_epochs: int = 30,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        if len(pretrained_path) > 0:
            self.load_pretrain_weights(pretrained_path)

    def load_pretrain_weights(self, pretrained_path):
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

    def get_patch_size(self):
        return self.net.patch_size

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_val_clim(self, clim):
        self.val_clim = clim

    def set_test_clim(self, clim):
        self.test_clim = clim

    def training_step(self, batch: Any, batch_idx: int):
        x, y, _, _, out_variables, region_info = batch
        loss_dict, _ = self.net.forward(x, y, out_variables, region_info, [lat_weighted_mse], lat=self.lat)
        loss_dict = loss_dict[0]
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        loss = loss_dict['loss']

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, _, variables, out_variables, region_info = batch
        pred_steps = 1
        pred_range = self.pred_range

        days = [int(pred_range / 24)]
        steps = [1]

        all_loss_dicts, _ = self.net.rollout(
            x,
            y,
            variables,
            out_variables,
            region_info,
            pred_steps,
            [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            self.denormalization,
            lat=self.lat,
            log_steps=steps,
            log_days=days,
            clim=self.val_clim
        )
        loss_dict = {}
        for d in all_loss_dicts:
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

    def test_step(self, batch: Any, batch_idx: int):
        x, y, _, variables, out_variables, region_info = batch
        pred_steps = 1
        pred_range = self.pred_range

        days = [int(pred_range / 24)]
        steps = [1]

        all_loss_dicts, _ = self.net.rollout(
            x,
            y,
            variables,
            out_variables,
            region_info,
            pred_steps,
            [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            self.denormalization,
            lat=self.lat,
            log_steps=steps,
            log_days=days,
            clim=self.test_clim
        )

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

        optimizer = torch.optim.Adam(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step', # or 'epoch'
            'frequency': 1
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

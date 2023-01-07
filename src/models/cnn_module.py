from typing import Any, Union

import torch
from numpy import isin
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from src.models.components.cnn_lstm import CNNLSTM
from src.models.components.resnet import ResNet
from src.models.components.unet import Unet
from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mean_bias,
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_nrmse,
    lat_weighted_rmse,
    mse,
    pearson,
)


class CNNLitModule(LightningModule):
    def __init__(
        self,
        net: Union[ResNet, Unet, CNNLSTM],
        pretrained_path: str,
        lr: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        weight_decay: float = 0.005,
        warmup_epochs: int = 5,
        max_epochs: int = 30,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        downscaling = False
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
        n, t, c, inp_h, inp_w = x.shape
        out_h, out_w = y.shape[-2], y.shape[-1]
        x = x.flatten(0, 1)
        x = torch.nn.functional.interpolate(x, (out_h, out_w), mode="bilinear")
        x = x.unflatten(0, sizes=(n, t))

        if self.net.climate_modeling:
            metric = [mse]
        else:
            metric = [lat_weighted_mse]

        loss_dict, _ = self.net.forward(x, y, out_variables, region_info, metric, lat=self.lat)
        loss_dict = loss_dict[0]
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        loss = loss_dict["loss"]

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, _, variables, out_variables, region_info = batch
        n, t, c, inp_h, inp_w = x.shape
        out_h, out_w = y.shape[-2], y.shape[-1]
        x = x.flatten(0, 1)
        x = torch.nn.functional.interpolate(x, (out_h, out_w), mode="bilinear")
        x = x.unflatten(0, sizes=(n, t))

        pred_steps = 1
        pred_range = self.pred_range

        days = [int(pred_range / 24)]
        steps = [1]

        if self.net.climate_modeling:
            metrics = [lat_weighted_mse_val, lat_weighted_rmse]
        elif self.hparams.downscaling:
            metrics = [lat_weighted_mse_val, lat_weighted_rmse, pearson, lat_weighted_mean_bias]
        else:
            metrics = [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc]

        all_loss_dicts, _ = self.net.rollout(
            x,
            y,
            variables,
            out_variables,
            region_info,
            pred_steps,
            metrics,
            self.denormalization,
            lat=self.lat,
            log_steps=steps,
            log_days=days,
            clim=self.val_clim,
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
        n, t, c, inp_h, inp_w = x.shape
        out_h, out_w = y.shape[-2], y.shape[-1]
        x = x.flatten(0, 1)
        x = torch.nn.functional.interpolate(x, (out_h, out_w), mode="bilinear")
        x = x.unflatten(0, sizes=(n, t))

        pred_steps = 1
        pred_range = self.pred_range

        days = [int(pred_range / 24)]
        steps = [1]

        if self.net.climate_modeling:
            metrics = [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_nrmse]
        elif self.hparams.downscaling:
            metrics = [lat_weighted_mse_val, lat_weighted_rmse, pearson, lat_weighted_mean_bias]
        else:
            metrics = [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc]

        all_loss_dicts, _ = self.net.rollout(
            x,
            y,
            variables,
            out_variables,
            region_info,
            pred_steps,
            metrics,
            self.denormalization,
            lat=self.lat,
            log_steps=steps,
            log_days=days,
            clim=self.test_clim,
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

        if self.net.climate_modeling and isinstance(self.net, CNNLSTM):
            optimizer = torch.optim.RMSprop(
                [
                    {
                        "params": decay,
                        "lr": self.hparams.lr,
                        "weight_decay": self.hparams.weight_decay,
                    },
                    {"params": no_decay, "lr": self.hparams.lr, "weight_decay": 0},
                ]
            )

            return {"optimizer": optimizer}
        else:
            optimizer = torch.optim.AdamW(
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
                        "weight_decay": 0,
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
            scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}  # or 'epoch'

            return {"optimizer": optimizer, "lr_scheduler": scheduler}

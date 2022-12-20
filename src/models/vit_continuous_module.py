# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any, Dict

import numpy as np
import torch
from pytorch_lightning import LightningModule
from src.models.components.tokenized_vit_continuous import \
    TokenizedViTContinuous
from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.metrics import (lat_weighted_acc, lat_weighted_mse,
                               lat_weighted_mse_val, lat_weighted_nrmse,
                               lat_weighted_rmse)
from src.utils.pos_embed import interpolate_pos_embed
from torchvision.transforms import transforms


def get_region_info(lat_range, lon_range, lat, lon, patch_size):
    lat = lat[::-1] # -90 to 90 from south (bottom) to north (top)
    h, w = len(lat), len(lon)
    lat_matrix = np.expand_dims(lat, axis=1).repeat(w, axis=1)
    lon_matrix = np.expand_dims(lon, axis=0).repeat(h, axis=0)
    valid_cells = (lat_matrix >= lat_range[0]) & (lat_matrix <= lat_range[1]) & (lon_matrix >= lon_range[0]) & (lon_matrix <= lon_range[1])
    h_ids, w_ids = np.nonzero(valid_cells)
    h_from, h_to = h_ids[0], h_ids[-1]
    w_from, w_to = w_ids[0], w_ids[-1]
    patch_idx = -1
    p = patch_size
    valid_patch_ids = []
    min_h, max_h = 1e5, -1e5
    min_w, max_w = 1e5, -1e5
    for i in range(0, h, p):
        for j in range(0, w, p):
            patch_idx += 1
            if (i >= h_from) & (i + p - 1 <= h_to) & (j >= w_from) & (j + p - 1 <= w_to):
                valid_patch_ids.append(patch_idx)
                min_h = min(min_h, i)
                max_h = max(max_h, i + p - 1)
                min_w = min(min_w, j)
                max_w = max(max_w, j + p - 1)
    return {
        'patch_ids': valid_patch_ids,
        'min_h': min_h,
        'max_h': max_h,
        'min_w': min_w,
        'max_w': max_w
    }


class ViTContinuousLitModule(LightningModule):
    def __init__(
        self,
        net: TokenizedViTContinuous,
        pretrained_path: str,
        lr: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        weight_decay: float = 0.005,
        warmup_epochs: int = 5,
        max_epochs: int = 30,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        if len(pretrained_path) > 0:
            self.load_mae_weights(pretrained_path)

    def load_mae_weights(self, pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))

        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

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
        # optimizer = self.optimizers()
        # optimizer.zero_grad()
        x, y, lead_times, variables, out_variables, region_info = batch
        loss_dict, _ = self.net.forward(x, y, lead_times, variables, out_variables, region_info, [lat_weighted_mse], lat=self.lat)
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

    # def training_step_end(self, step_output):
    #     lr_scheduler = self.lr_schedulers()
    #     lr_scheduler.step()

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables, region_info = batch
        pred_steps = 1
        pred_range = self.pred_range

        days = [int(pred_range / 24)]
        steps = [1]

        if self.net.climate_modeling:
            metrics = [lat_weighted_mse_val, lat_weighted_rmse]
        else:
            metrics = [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc]

        all_loss_dicts, _ = self.net.rollout(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            region_info,
            pred_steps,
            metrics,
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

    # def validation_epoch_end(self, outputs: List[Any]):
    #     acc = self.val_acc.compute()  # get val accuracy from current epoch
    #     self.val_acc_best.update(acc)
    #     self.log(
    #         "val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True
    #     )

    #     self.val_acc.reset()  # reset val accuracy for next epoch

    def test_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables, region_info = batch
        pred_steps = 1
        pred_range = self.pred_range

        days = [int(pred_range / 24)]
        steps = [1]

        if self.net.climate_modeling:
            metrics = [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_nrmse]
        else:
            metrics = [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc]

        all_loss_dicts, _ = self.net.rollout(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            region_info,
            pred_steps,
            metrics,
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
                prog_bar=False,
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

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

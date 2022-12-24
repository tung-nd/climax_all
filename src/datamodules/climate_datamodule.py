import os
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from datamodules import BOUNDARIES

from .climate_dataset import ClimateBenchDataset, load_x_y, split_train_val


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    lead_times = torch.cat([batch[i][2] for i in range(len(batch))])
    variables = batch[0][3]
    out_variables = batch[0][4]
    region_info = batch[0][5]
    return (
        inp,
        out,
        lead_times,
        variables,
        out_variables,
        region_info,
    )


class ClimateDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,  # contains metadata and train + val + test
        history=10,
        list_train_simu=[
            'ssp126',
            'ssp370',
            'ssp585',
            'historical',
            'hist-GHG',
            'hist-aer'
        ],
        list_test_simu=[
            'ssp245'
        ],
        variables=[
            'CO2',
            'SO2',
            'CH4',
            'BC'
        ],
        out_variables='tas',
        train_ratio=0.9,
        region: str = 'Global',
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        if isinstance(out_variables, str):
            out_variables = [out_variables]
            self.hparams.out_variables = out_variables

        x_train, y_train, lat, lon = load_x_y(os.path.join(root_dir, 'train_val'), list_train_simu, out_variables)
        self.lat, self.lon = lat, lon
        # x_train, y_train, x_val, y_val = split_train_val(x_train_val, y_train_val, train_ratio)
        x_test, y_test, _, _ = load_x_y(os.path.join(root_dir, 'test'), list_test_simu, out_variables)

        self.dataset_train = ClimateBenchDataset(
            x_train, y_train, history, variables, out_variables, lat, 'train'
        )
        
        # self.dataset_val = ClimateBenchDataset(
        #     x_val, y_val, variables, out_variables, lat, 'val'
        # )
        # self.dataset_val.set_normalize(self.dataset_train.inp_transform, self.dataset_train.out_transform)

        self.dataset_test = ClimateBenchDataset(
            x_test, y_test, history, variables, out_variables, lat, 'test'
        )
        self.dataset_test.set_normalize(self.dataset_train.inp_transform, self.dataset_train.out_transform)

    def get_region_info(self, region):
        region = BOUNDARIES[region]
        lat_range = region['lat_range']
        lon_range = region['lon_range']
        lat, lon = self.get_lat_lon()
        lat = lat[::-1] # -90 to 90 from south (bottom) to north (top)
        h, w = len(lat), len(lon)
        lat_matrix = np.expand_dims(lat, axis=1).repeat(w, axis=1)
        lon_matrix = np.expand_dims(lon, axis=0).repeat(h, axis=0)
        valid_cells = (lat_matrix >= lat_range[0]) & (lat_matrix <= lat_range[1]) & (lon_matrix >= lon_range[0]) & (lon_matrix <= lon_range[1])
        h_ids, w_ids = np.nonzero(valid_cells)
        h_from, h_to = h_ids[0], h_ids[-1]
        w_from, w_to = w_ids[0], w_ids[-1]
        patch_idx = -1
        p = self.patch_size
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

    def get_lat_lon(self):
        return self.lat, self.lon

    def set_patch_size(self, p):
        self.patch_size = p

    def get_test_clim(self):
        return self.dataset_test.y_normalization

    def setup(self, stage: Optional[str] = None):
        region_info = self.get_region_info(self.hparams.region)
        self.dataset_train.set_region_info(region_info)
        # self.dataset_val.set_region_info(region_info)
        self.dataset_test.set_region_info(region_info)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            # drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.dataset_val,
    #         batch_size=self.hparams.batch_size,
    #         shuffle=False,
    #         # drop_last=True,
    #         num_workers=self.hparams.num_workers,
    #         pin_memory=self.hparams.pin_memory,
    #         collate_fn=collate_fn,
    #     )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            # drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

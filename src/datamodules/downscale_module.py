import os
from typing import Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

from datamodules import BOUNDARIES, VAR_LEVEL_TO_NAME_LEVEL

from .finetune_iterdataset import (
    Forecast,
    IndividualForecastDataIter,
    NpyReader,
    ShuffleIterableDataset,
)


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    lead_times = torch.stack([batch[i][2] for i in range(len(batch))])
    variables = batch[0][3]
    out_variables = batch[0][4]
    region_info = batch[0][5]
    return (
        inp,
        out,
        lead_times,
        [VAR_LEVEL_TO_NAME_LEVEL[v] for v in variables],
        [VAR_LEVEL_TO_NAME_LEVEL[v] for v in out_variables],
        region_info,
    )


class ERA5DownscaleModule(LightningDataModule):
    def __init__(
        self,
        root_dir,  # contains metadata and train + val + test
        variables,
        buffer_size,
        out_variables=None,
        region: str = 'Global',
        subsample: int = 1,
        pct_train: float = 1.0,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        if isinstance(out_variables, str):
            out_variables = [out_variables]
            self.hparams.out_variables = out_variables

        self.lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
        if pct_train < 1.0:
            train_len = int(pct_train * len(self.lister_train))
            self.lister_train = self.lister_train[:train_len]
        self.lister_val = None
        self.lister_test = None
        if os.path.exists(os.path.join(root_dir, "val")):
            self.lister_val = list(dp.iter.FileLister(os.path.join(root_dir, "val")))
            self.lister_test = list(dp.iter.FileLister(os.path.join(root_dir, "test")))

        self.transforms = self.get_normalize(type='input')
        self.output_transforms = self.get_normalize(variables=out_variables, type='output')

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None

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

    def get_normalize(self, type, variables=None):
        assert type in ['input', 'output']
        if variables is None:
            variables = self.hparams.variables
        normalize_mean = dict(np.load(os.path.join(self.hparams.root_dir, f"normalize_mean_{type}.npz")))
        mean = []
        for var in variables:
            if var != "tp":
                mean.append(normalize_mean[VAR_LEVEL_TO_NAME_LEVEL[var]])
            else:
                mean.append(np.array([0.0]))
        normalize_mean = np.concatenate(mean)
        normalize_std = dict(np.load(os.path.join(self.hparams.root_dir, f"normalize_std_{type}.npz")))
        normalize_std = np.concatenate(
            [normalize_std[VAR_LEVEL_TO_NAME_LEVEL[var]] for var in variables]
        )
        return transforms.Normalize(normalize_mean, normalize_std)

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npy"))
        return lat, lon

    def set_patch_size(self, p):
        self.patch_size = p

    def setup(self, stage: Optional[str] = None):
        region_info = self.get_region_info(self.hparams.region)
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ShuffleIterableDataset(
                IndividualForecastDataIter(
                    Forecast(
                        NpyReader(
                            file_list=self.lister_train,
                            variables=self.hparams.variables,
                            out_variables=self.hparams.out_variables,
                            shuffle=True,
                        ),
                        predict_range=0,
                        hrs_each_step=1, # does not matter
                        subsample=self.hparams.subsample
                    ),
                    transforms=self.transforms,
                    output_transforms=self.output_transforms,
                    region_info=region_info
                ),
                buffer_size=self.hparams.buffer_size,
            )

            if self.lister_val is not None:
                self.data_val = IndividualForecastDataIter(
                    Forecast(
                        NpyReader(
                            file_list=self.lister_val,
                            variables=self.hparams.variables,
                            out_variables=self.hparams.out_variables,
                        ),
                        predict_range=0,
                        hrs_each_step=1, # does not matter
                        subsample=self.hparams.subsample
                    ),
                    transforms=self.transforms,
                    output_transforms=self.output_transforms,
                    region_info=region_info
                )

            if self.lister_test is not None:
                self.data_test = IndividualForecastDataIter(
                    Forecast(
                        NpyReader(
                            file_list=self.lister_test,
                            variables=self.hparams.variables,
                            out_variables=self.hparams.out_variables,
                        ),
                        predict_range=0,
                        hrs_each_step=1, # does not matter
                        subsample=self.hparams.subsample
                    ),
                    transforms=self.transforms,
                    output_transforms=self.output_transforms,
                    region_info=region_info
                )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            # shuffle=True,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        if self.lister_val is not None:
            return DataLoader(
                self.data_val,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=collate_fn,
            )

    def test_dataloader(self):
        if self.lister_test is not None:
            return DataLoader(
                self.data_test,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=collate_fn,
            )


# era5 = ERA5IterDatasetModule(
#     "/datadrive/datasets/5.625deg_equally_np/",
#     "npy",
#     "image",
#     ["t2m", "u10", "v10", "z_850", "z_500"],
#     1000,
#     batch_size=64,
#     num_workers=2,
#     pin_memory=False,
# )
# era5.setup()
# for x, variables in era5.train_dataloader():
#     print(x.shape)
#     print (variables)
#     break

# era5 = ERA5IterDatasetModule(
#     "/datadrive/datasets/5.625deg_equally_np/",
#     "npy",
#     "video",
#     ["t2m", "u10", "v10", "z"],
#     1000,
#     batch_size=64,
#     num_workers=2,
#     pin_memory=False,
# )
# era5.setup()
# for x, variables in era5.train_dataloader():
#     print(x.shape)
#     print (variables)
#     break

# era5 = ERA5IterDatasetModule(
#     "/datadrive/datasets/5.625deg_equally_np/",
#     "npy",
#     "forecast",
#     ["t2m", "u10", "v10", "z_850", "z_500"],
#     1000,
#     out_variables=["t2m", "u10", "v10", "z_850", "z_500"],
#     batch_size=64,
#     num_workers=2,
#     pin_memory=False,
# )
# era5.setup()
# for x, y, variables, out_variables in era5.train_dataloader():
#     print(x.shape)
#     print(y.shape)
#     print (era5.transforms)
#     print ('mean input channel', x.mean(dim=(0, 1, 3, 4)))
#     print ('std input channel', x.std(dim=(0, 1, 3, 4)))
#     # print (era5.output_transforms)
#     # print ('mean output channel', y.mean(dim=(0, 2, 3)))
#     # print ('std output channel', y.std(dim=(0, 2, 3)))
#     print (variables)
#     # print (out_variables)
#     break
# for x, y, variables, out_variables in era5.val_dataloader():
#     print(x.shape)
#     print(y.shape)
#     # print (era5.transforms)
#     # print ('mean input channel', x.mean(dim=(0, 1, 3, 4)))
#     # print ('std input channel', x.std(dim=(0, 1, 3, 4)))
#     print (era5.output_transforms)
#     print ('mean output channel', y.mean(dim=(0, 1, 3, 4)))
#     print ('std output channel', y.std(dim=(0, 1, 3, 4)))
#     # print (variables)
#     print (out_variables)
#     break

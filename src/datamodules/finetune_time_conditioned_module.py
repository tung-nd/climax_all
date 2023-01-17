import os
from typing import List, Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

from datamodules import BOUNDARIES, VAR_LEVEL_TO_NAME_LEVEL

from .pretrain_iterdataset import (
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
        region_info
    )


class FinetuneTimeConditionedModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        variables,
        buffer_size,
        out_variables = None,
        region = 'Global',
        max_predict_range = 168,
        random_lead_time = True,
        hrs_each_step = 1,
        history = 1,
        interval = 0,
        subsample = 1,
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
        if out_variables is None:
            self.hparams.out_variables = variables

        self.lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
        self.lister_val = list(dp.iter.FileLister(os.path.join(root_dir, "val")))

        self.dataset_args = {
            "max_predict_range": max_predict_range,
            "random_lead_time": random_lead_time,
            "hrs_each_step": hrs_each_step,
            "history": history,
            "interval": interval,
            "subsample": subsample,
        }

        self.transforms = self.get_normalize()
        self.output_transforms = self.get_normalize(self.hparams.out_variables)

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None

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

    def get_normalize(self, variables=None):
        if variables is None:
            variables = self.hparams.variables
        normalize_mean = dict(np.load(os.path.join(self.hparams.root_dir, "normalize_mean.npz")))
        mean = []
        for var in variables:
            if var != "tp":
                mean.append(normalize_mean[VAR_LEVEL_TO_NAME_LEVEL[var]])
            else:
                mean.append(np.array([0.0]))
        normalize_mean = np.concatenate(mean)
        normalize_std = dict(np.load(os.path.join(self.hparams.root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate(
            [normalize_std[VAR_LEVEL_TO_NAME_LEVEL[var]] for var in variables]
        )
        return transforms.Normalize(normalize_mean, normalize_std)

    def get_lat_lon(self):
        # assume different data sources have the same lat and lon coverage
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npy"))
        return lat, lon

    def set_patch_size(self, p):
        self.patch_size = p

    def setup(self, stage: Optional[str] = None):
        region_info = self.get_region_info(self.hparams.region)
        dataset_args = self.dataset_args
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val:
            self.data_train = ShuffleIterableDataset(
                IndividualForecastDataIter(
                    Forecast(
                        NpyReader(
                            self.lister_train,
                            start_idx=0,
                            end_idx=1,
                            variables=self.hparams.variables,
                            out_variables=self.hparams.out_variables,
                            shuffle=True,
                            multi_dataset_training=False
                        ),
                        **dataset_args,
                    ),
                    self.transforms,
                    self.output_transforms,
                    region_info=region_info
                ),
                buffer_size=self.hparams.buffer_size,
            )

            self.data_val = IndividualForecastDataIter(
                Forecast(
                    NpyReader(
                        self.lister_val,
                        start_idx=0,
                        end_idx=1,
                        variables=self.hparams.variables,
                        out_variables=self.hparams.out_variables,
                        shuffle=False,
                        multi_dataset_training=False
                    ),
                    **dataset_args,
                ),
                self.transforms,
                self.output_transforms,
                region_info=region_info
            )


    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,            
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


# dataset_type = 'forecast'
# dict_root_dirs = {
#     'mpi-esm': '/datadrive/datasets/CMIP6/MPI-ESM/5.625deg_equally_np_all_levels',
#     'taiesm': '/datadrive/datasets/CMIP6/TaiESM1/5.625deg_equally_np_all_levels'
# }
# dict_buffer_sizes = {'mpi-esm': 1000, 'taiesm': 1000}
# dict_in_variables = {
#     'mpi-esm': ['t2m', 'z_500', 't_850'],
#     'taiesm': ['z_500', 't_850']
# }
# dict_out_variables = {
#     'mpi-esm': ['z_500', 't_850'],
#     'taiesm': ['z_500', 't_850']
# }
# dict_predict_ranges = {'mpi-esm': 12, 'taiesm': 12}
# dict_histories = {'mpi-esm': 1, 'taiesm': 1}
# dict_intervals = {'mpi-esm': 0, 'taiesm': 0}
# dict_subsamples = {'mpi-esm': 1, 'taiesm': 1}

# datamodule = MultiSourceTrainDatasetModule(
#     dataset_type,
#     dict_root_dirs,
#     dict_buffer_sizes,
#     dict_in_variables,
#     dict_out_variables,
#     dict_predict_ranges=dict_predict_ranges,
#     dict_histories=dict_histories,
#     dict_intervals=dict_intervals,
#     dict_subsamples=dict_subsamples,
#     batch_size=16,
#     num_workers=1,
#     pin_memory=False
# )
# datamodule.setup()
# dataloader = datamodule.train_dataloader()
# for batch in dataloader:
#     for k in batch.keys():
#         print (k)
#         x1, y1, in1, out1 = batch[k]
#         print (x1.shape)
#         print (y1.shape)
#         print (in1)
#         print (out1)
#     break

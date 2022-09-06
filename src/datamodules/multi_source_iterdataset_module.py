import os
from typing import List, Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

from src.datamodules import VAR_LEVEL_TO_NAME_LEVEL

from era5_iterdataset import (
    ERA5Npy,
    ERA5,
    ERA5Video,
    ERA5Forecast,
    IndividualDataIter,
    IndividualForecastDataIter,
    ShuffleIterableDataset,
)


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    variables = batch[0][1]
    return inp, [VAR_LEVEL_TO_NAME_LEVEL[v] for v in variables]


def collate_forecast_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    variables = batch[0][2]
    out_variables = batch[0][3]
    return (
        inp,
        out,
        [VAR_LEVEL_TO_NAME_LEVEL[v] for v in variables],
        [VAR_LEVEL_TO_NAME_LEVEL[v] for v in out_variables],
    )


def collate_forecast_precip_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    tp = torch.stack([batch[i][2] for i in range(len(batch))])
    variables = batch[0][3]
    return inp, out, tp, [VAR_LEVEL_TO_NAME_LEVEL[v] for v in variables]


class MultiSourceTrainDatasetModule(LightningDataModule):
    def __init__(
        self,
        dataset_type: str,  # image, video, or forecast (finetune)
        list_root_dirs: List[str],  # list of root dirs
        list_buffer_sizes: List[int], # list of buffer sizes 
        list_in_variables: List[List], # list of lists of input variables
        list_out_variables: List[List],
        list_timesteps: List[int] = [2],  # only used for video
        list_predict_ranges: List[int] = [72],  # only used for forecast
        list_histories: List[int] = [1],  # used for forecast
        list_intervals: List[int] = [12],  # used for forecast and video
        list_subsamples: List[int] = [1],  # used for forecast
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        out_variables = []
        for i, list_out in enumerate(list_out_variables):
            if list_out is not None:
                out_variables.append(list_out)
            else:
                out_variables.append(list_in_variables[i])
        self.hparams.list_out_variables = out_variables

        self.reader = ERA5Npy
        self.list_lister_trains = [list(dp.iter.FileLister(os.path.join(root_dir, "train"))) for root_dir in list_root_dirs]

        if dataset_type == "image":
            self.train_dataset_class = ERA5
            self.train_dataset_args = [{} for _ in range(len(list_root_dirs))]
            self.data_iter = IndividualDataIter
            self.collate_fn = collate_fn
        elif dataset_type == "video":
            self.train_dataset_class = ERA5Video
            self.train_dataset_args = [{"timesteps": timesteps, "interval": interval} for timesteps, interval in zip(list_timesteps, list_intervals)]
            self.data_iter = IndividualDataIter
            self.collate_fn = collate_fn
        elif dataset_type == "forecast":
            self.train_dataset_class = ERA5Forecast
            self.train_dataset_args = [{
                "predict_range": predict_range,
                "history": history,
                "interval": interval,
                "subsample": subsample,
            } for predict_range, history, interval, subsample in zip(list_predict_ranges, list_histories, list_intervals, list_subsamples)]
            self.data_iter = IndividualForecastDataIter
            self.collate_fn = collate_forecast_fn
        else:
            raise NotImplementedError("Only support image, video, or forecast dataset")

        self.transforms = self.get_normalize()
        self.output_transforms = self.get_normalize(list_out_variables)

        self.list_data_train: Optional[List[IterableDataset]] = None

    def get_normalize(self, list_variables=None):
        if list_variables is None:
            list_variables = self.hparams.list_in_variables
        list_transforms = []
        for root_dir, variables in zip(self.hparams.list_root_dirs, list_variables):
            normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
            mean = []
            for var in variables:
                if var != "tp":
                    mean.append(normalize_mean[VAR_LEVEL_TO_NAME_LEVEL[var]])
                else:
                    mean.append(np.array([0.0]))
            normalize_mean = np.concatenate(mean)
            normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
            normalize_std = np.concatenate(
                [normalize_std[VAR_LEVEL_TO_NAME_LEVEL[var]] for var in variables]
            )
            list_transforms.append(transforms.Normalize(normalize_mean, normalize_std))
        return list_transforms

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.list_root_dirs[0], "lat.npy"))
        lon = np.load(os.path.join(self.hparams.list_root_dirs[0], "lon.npy"))
        return lat, lon

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.list_data_train:
            list_data_train = []
            for i in range(len(self.list_lister_trains)):
                lister_train = self.list_lister_trains[i]
                variables = self.hparams.list_in_variables[i]
                out_variables = self.hparams.list_out_variables[i]
                dataset_args = self.train_dataset_args[i]
                transforms = self.transforms[i]
                output_transforms = self.output_transforms[i]
                buffer_size = self.hparams.list_buffer_sizes[i]
                list_data_train.append(
                    ShuffleIterableDataset(
                        self.data_iter(
                            self.train_dataset_class(
                                self.reader(
                                    lister_train,
                                    variables=variables,
                                    out_variables=out_variables,
                                    shuffle=True,
                                ),
                                **dataset_args,
                            ),
                            transforms,
                            output_transforms,
                        ),
                        buffer_size,
                    )
                )
            self.list_data_train = list_data_train

    def train_dataloader(self):
        loaders = [DataLoader(
            data_train,
            batch_size=self.hparams.batch_size,
            # shuffle=True,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        ) for data_train in self.list_data_train]
        return CombinedLoader(loaders, mode="max_size_cycle")

dataset_type = 'forecast'  # image, video, or forecast (finetune)
list_root_dirs = ['/datadrive/datasets/CMIP6/MPI-ESM/5.625deg_equally_np_all_levels', '/datadrive/datasets/CMIP6/TaiESM1/5.625deg_equally_np_all_levels/']  # list of root dirs
list_buffer_sizes = [1000, 1000] # list of buffer sizes 
list_in_variables = [['t2m', 'z_500', 't_850'], ['z_500', 't_850']] # list of lists of input variables
list_out_variables = [['z_500', 't_850'], ['z_500', 't_850']]
# list_timesteps = [2, 2]  # only used for video
list_predict_ranges = [12, 12]  # only used for forecast
list_histories = [1, 1]  # used for forecast
list_intervals = [0, 0]  # used for forecast and video
list_subsamples = [1, 1]  # used for forecast

datamodule = MultiSourceTrainDatasetModule(
    dataset_type,
    list_root_dirs,
    list_buffer_sizes,
    list_in_variables,
    list_out_variables,
    list_predict_ranges=list_predict_ranges,
    list_histories=list_histories,
    list_intervals=list_intervals,
    list_subsamples=list_subsamples,
    batch_size=16,
    num_workers=1,
    pin_memory=False
)
datamodule.setup()
dataloader = datamodule.train_dataloader()
for x, y in dataloader:
    print (len(x))
    print (len(y))
    break
    # print (vars)
    # print (out_vars)

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

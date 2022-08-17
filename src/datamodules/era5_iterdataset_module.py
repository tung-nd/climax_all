import os
from typing import Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

from datamodules import VAR_LEVEL_TO_NAME_LEVEL

from .era5_iterdataset import (
    ERA5,
    ERA5Forecast,
    ERA5ForecastMultiStep,
    ERA5ForecastMultiStepPrecip,
    ERA5ForecastPrecip,
    ERA5Npy,
    ERA5Video,
    IndividualDataIter,
    IndividualForecastDataIter,
    IndividualForecastPrecipDataIter,
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
    return inp, out, [VAR_LEVEL_TO_NAME_LEVEL[v] for v in variables]


def collate_forecast_precip_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    tp = torch.stack([batch[i][2] for i in range(len(batch))])
    variables = batch[0][3]
    return inp, out, tp, [VAR_LEVEL_TO_NAME_LEVEL[v] for v in variables]


class ERA5IterDatasetModule(LightningDataModule):
    def __init__(
        self,
        root_dir,  # contains metadata and train + val + test
        reader,  # npy or zarr
        dataset_type,  # image, video, or forecast (finetune)
        variables,
        buffer_size,
        timesteps: int = 8,  # only used for video
        predict_range: int = 6,  # only used for forecast
        predict_steps: int = 4,  # only used for forecast
        pct_train: float = 1.0,  # percentage of data used for training
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        if reader == "npy":
            self.reader = ERA5Npy
            self.lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
            if pct_train < 1.0:
                train_len = int(pct_train * len(self.lister_train))
                self.lister_train = self.lister_train[:train_len]
            self.lister_val = None
            self.lister_test = None
            if os.path.exists(os.path.join(root_dir, "val")):
                self.lister_val = list(dp.iter.FileLister(os.path.join(root_dir, "val")))
                self.lister_test = list(dp.iter.FileLister(os.path.join(root_dir, "test")))
        else:
            raise NotImplementedError(f"Only support npy or zarr")

        if dataset_type == "image":
            self.train_dataset_class = ERA5
            self.train_dataset_args = {}
            self.val_dataset_class = ERA5
            self.val_dataset_args = {}
            self.data_iter = IndividualDataIter
            self.collate_fn = collate_fn
        elif dataset_type == "video":
            self.train_dataset_class = ERA5Video
            self.train_dataset_args = {"timesteps": timesteps}
            self.val_dataset_class = ERA5Video
            self.val_dataset_args = {"timesteps": timesteps}
            self.data_iter = IndividualDataIter
            self.collate_fn = collate_fn
        elif dataset_type == "forecast":
            self.train_dataset_class = ERA5Forecast
            self.train_dataset_args = {"predict_range": predict_range}
            self.val_dataset_class = ERA5ForecastMultiStep
            self.val_dataset_args = {
                "pred_range": predict_range,
                "pred_steps": predict_steps,
            }
            self.data_iter = IndividualForecastDataIter
            self.collate_fn = collate_forecast_fn
        elif dataset_type == "forecast_precip":
            self.train_dataset_class = ERA5ForecastPrecip
            self.train_dataset_args = {"predict_range": predict_range}
            self.val_dataset_class = ERA5ForecastMultiStepPrecip
            self.val_dataset_args = {
                "pred_range": predict_range,
                "pred_steps": predict_steps,
            }
            self.data_iter = IndividualForecastPrecipDataIter
            self.collate_fn = collate_forecast_precip_fn
        else:
            raise NotImplementedError("Only support image, video, or forecast dataset")

        self.transforms = self.get_normalize()

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None

    def get_normalize(self):
        normalize_mean = dict(np.load(os.path.join(self.hparams.root_dir, "normalize_mean.npz")))
        normalize_mean = np.concatenate(
            [normalize_mean[VAR_LEVEL_TO_NAME_LEVEL[var]] for var in self.hparams.variables if var != "tp"]
        )
        normalize_std = dict(np.load(os.path.join(self.hparams.root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate(
            [normalize_std[VAR_LEVEL_TO_NAME_LEVEL[var]] for var in self.hparams.variables if var != "tp"]
        )
        return transforms.Normalize(normalize_mean, normalize_std)

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npy"))
        return lat, lon

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ShuffleIterableDataset(
                self.data_iter(
                    self.train_dataset_class(
                        self.reader(
                            self.lister_train,
                            variables=self.hparams.variables,
                            shuffle=True,
                        ),
                        **self.train_dataset_args,
                    ),
                    self.transforms,
                ),
                self.hparams.buffer_size,
            )

            if self.lister_val is not None:
                self.data_val = self.data_iter(
                    self.val_dataset_class(
                        self.reader(
                            self.lister_val,
                            variables=self.hparams.variables,
                        ),
                        **self.val_dataset_args,
                    ),
                    self.transforms,
                )

            if self.lister_test is not None:
                self.data_test = self.data_iter(
                    self.val_dataset_class(
                        self.reader(
                            self.lister_test,
                            variables=self.hparams.variables,
                        ),
                        **self.val_dataset_args,
                    ),
                    self.transforms,
                )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            # shuffle=True,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.lister_val is not None:
            return DataLoader(
                self.data_val,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collate_fn,
            )

    def test_dataloader(self):
        if self.lister_test is not None:
            return DataLoader(
                self.data_test,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collate_fn,
            )


# era5 = ERA5IterDatasetModule(
#     "/datadrive/5.625deg_equally_np/",
#     "npy",
#     "image",
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
#     "/datadrive/5.625deg_equally_np/",
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
#     "/datadrive/5.625deg_equally_np/",
#     "npy",
#     "forecast",
#     ["t2m", "u10", "v10", "z"],
#     1000,
#     batch_size=64,
#     num_workers=2,
#     pin_memory=False,
# )
# era5.setup()
# for x, y, variables in era5.train_dataloader():
#     print(x.shape)
#     print(y.shape)
#     print (variables)
#     break
# for x, y, variables in era5.val_dataloader():
#     print(x.shape)
#     print(y.shape)
#     print (variables)
#     break

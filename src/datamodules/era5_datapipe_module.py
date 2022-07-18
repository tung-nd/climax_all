import glob
import os
from typing import Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datamodules import VAR_TO_NAME

from .era5_datapipe import (
    ERA5,
    ERA5Forecast,
    ERA5Npy,
    ERA5Video,
    ERA5Zarr,
    IndividualDataIter,
    IndividualForecastDataIter,
)


def collate_fn(batch):
    inp = torch.stack([batch[i] for i in range(len(batch))])
    return inp


def collate_forecast_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    return inp, out


class ERA5DataPipeModule(LightningDataModule):
    def __init__(
        self,
        root_dir,  # contains metadata and train + val + test
        reader,  # npy or zarr
        dataset_type,  # image, video, or forecast (finetune)
        variables,
        buffer_size,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        if reader == "npy":
            self.reader = ERA5Npy
            self.lister_train = dp.iter.FileLister(os.path.join(root_dir, "train"))
            self.lister_val = dp.iter.FileLister(os.path.join(root_dir, "val"))
            self.lister_test = dp.iter.FileLister(os.path.join(root_dir, "test"))
        elif reader == "zarr":
            self.reader = ERA5Zarr
            self.lister_train = dp.iter.IterableWrapper(
                glob.glob(os.path.join(root_dir, "train", "*.zarr"))
            )
            self.lister_val = dp.iter.IterableWrapper(
                glob.glob(os.path.join(root_dir, "val", "*.zarr"))
            )
            self.lister_test = dp.iter.IterableWrapper(
                glob.glob(os.path.join(root_dir, "test", "*.zarr"))
            )
        else:
            raise NotImplementedError(f"Only support npy or zarr")

        if dataset_type == "image":
            self.dataset_class = ERA5
            self.data_iter = IndividualDataIter
            self.collate_fn = collate_fn
        elif dataset_type == "video":
            self.dataset_class = ERA5Video
            self.data_iter = IndividualDataIter
            self.collate_fn = collate_fn
        elif dataset_type == "forecast":
            self.dataset_class = ERA5Forecast
            self.data_iter = IndividualForecastDataIter
            self.collate_fn = collate_forecast_fn
        else:
            raise NotImplementedError("Only support image, video, or forecast dataset")

        self.transforms = self.get_normalize()

        self.data_train: Optional[dp.iter.IterDataPipe] = None
        self.data_val: Optional[dp.iter.IterDataPipe] = None
        self.data_test: Optional[dp.iter.IterDataPipe] = None

    def get_normalize(self):
        normalize_mean = dict(
            np.load(os.path.join(self.hparams.root_dir, "normalize_mean.npz"))
        )
        normalize_mean = np.concatenate(
            [normalize_mean[VAR_TO_NAME[var]] for var in self.hparams.variables]
        )
        normalize_std = dict(
            np.load(os.path.join(self.hparams.root_dir, "normalize_std.npz"))
        )
        normalize_std = np.concatenate(
            [normalize_std[VAR_TO_NAME[var]] for var in self.hparams.variables]
        )
        return transforms.Normalize(normalize_mean, normalize_std)

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = (
                self.data_iter(
                    self.dataset_class(
                        self.reader(
                            self.lister_train.shuffle().sharding_filter(),
                            variables=self.hparams.variables,
                        )
                    ),
                    self.transforms,
                )
                .shuffle(
                    buffer_size=self.hparams.buffer_size
                )  # shuffle at the individual data level
                .batch(self.hparams.batch_size)
                .in_batch_shuffle()  # shuffle within a batch, probably not necessary
                .collate(self.collate_fn)
            )

            self.data_val = (
                self.data_iter(
                    self.dataset_class(
                        self.reader(self.lister_val, variables=self.hparams.variables,)
                    ),
                    self.transforms,
                )
                .batch(self.hparams.batch_size)
                .collate(self.collate_fn)
            )

            self.data_test = (
                self.data_iter(
                    self.dataset_class(
                        self.reader(self.lister_test, variables=self.hparams.variables,)
                    ),
                    self.transforms,
                )
                .batch(self.hparams.batch_size)
                .collate(self.collate_fn)
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )


# era5 = ERA5DataPipeModule(
#     "/datadrive/datasets/1.40625deg_monthly_np/",
#     "npy",
#     "image",
#     ["t2m", "z", "t"],
#     1000,
#     64,
#     2,
#     False,
# )
# era5.setup()
# for x in era5.train_dataloader():
#     print(x.shape)
#     print(x.mean(dim=(0, 2, 3)))
#     print(x.std(dim=(0, 2, 3)))
#     break

# era5 = ERA5DataPipeModule(
#     "/mnt/weatherbench/tmp/_yearly_np",
#     "npy",
#     "video",
#     ["t2m", "z", "t"],
#     1000,
#     16,
#     0,
#     False,
# )
# era5.setup()
# for x in era5.train_dataloader():
#     print(x.shape)
#     print(x.mean(dim=(0, 2, 3)))
#     print(x.std(dim=(0, 2, 3)))
#     break

# era5 = ERA5DataPipeModule(
#     "/mnt/weatherbench/tmp/_yearly_np",
#     "npy",
#     "forecast",
#     ["t2m", "z", "t"],
#     1000,
#     64,
#     2,
#     False,
# )
# era5.setup()
# for x, y in era5.train_dataloader():
#     print(x.shape)
#     print(x.mean(dim=(0, 2, 3)))
#     print(x.std(dim=(0, 2, 3)))
#     print(y.shape)
#     print(y.mean(dim=(0, 2, 3)))
#     print(y.std(dim=(0, 2, 3)))
#     break

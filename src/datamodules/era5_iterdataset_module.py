import glob
import os
from typing import Iterable, Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

from datamodules import VAR_TO_NAME

from .era5_iterdataset import (
    ERA5,
    ERA5Forecast,
    ERA5ForecastMultiStep,
    ERA5Npy,
    ERA5Video,
    IndividualDataIter,
    IndividualForecastDataIter,
    ShuffleIterableDataset,
)


def collate_fn(batch):
    inp = torch.stack([batch[i] for i in range(len(batch))])
    return inp


def collate_forecast_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    return inp, out


class ERA5IterDatasetModule(LightningDataModule):
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
            self.lister_train = list(
                dp.iter.FileLister(os.path.join(root_dir, "train"))
            )
            self.lister_val = list(dp.iter.FileLister(os.path.join(root_dir, "val")))
            self.lister_test = list(dp.iter.FileLister(os.path.join(root_dir, "test")))
        else:
            raise NotImplementedError(f"Only support npy or zarr")

        if dataset_type == "image":
            self.train_dataset_class = ERA5
            self.val_dataset_class = ERA5
            self.data_iter = IndividualDataIter
            self.collate_fn = collate_fn
        elif dataset_type == "video":
            self.train_dataset_class = ERA5Video
            self.val_dataset_class = ERA5Video
            self.data_iter = IndividualDataIter
            self.collate_fn = collate_fn
        elif dataset_type == "forecast":
            self.train_dataset_class = ERA5Forecast
            self.val_dataset_class = ERA5ForecastMultiStep
            self.data_iter = IndividualForecastDataIter
            self.collate_fn = collate_forecast_fn
        else:
            raise NotImplementedError("Only support image, video, or forecast dataset")

        self.transforms = self.get_normalize()

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None

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
            self.data_train = ShuffleIterableDataset(
                self.data_iter(
                    self.train_dataset_class(
                        self.reader(
                            self.lister_train,
                            variables=self.hparams.variables,
                            shuffle=True,
                        )
                    ),
                    self.transforms,
                ),
                self.hparams.buffer_size,
            )

            self.data_val = self.data_iter(
                self.val_dataset_class(
                    self.reader(self.lister_val, variables=self.hparams.variables,)
                ),
                self.transforms,
            )

            self.data_test = self.data_iter(
                self.val_dataset_class(
                    self.reader(self.lister_test, variables=self.hparams.variables,)
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
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )


# era5 = ERA5IterDatasetModule(
#     "/datadrive/1.40625deg_equally_np/",
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
#     print(y.shape)
#     break
# for x, y in era5.val_dataloader():
#     print(x.shape)
#     print(y.shape)
#     break

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

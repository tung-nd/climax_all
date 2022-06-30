from typing import List, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from src.datamodules.era5_surface import ERA5Surface, ERA5SurfaceForecast
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms


class ERA5SurfaceDataModule(LightningDataModule):
    def __init__(
        self,
        variable_data_paths: List = [
            '/mnt/weatherbench/temperature_2m.npy',
            '/mnt/weatherbench/wind_u_10m.npy',
            '/mnt/weatherbench/wind_v_10m.npy',
        ],
        train_val_test_split: Tuple[int, int, int] = (300000, 25000, 25640),
        normalize = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Normalize(mean=[277.0595, -0.05025468, 0.18755548], std=[21.289722, 5.5454874, 4.764006]) if normalize else None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = ERA5Surface(self.hparams.variable_data_paths, self.transforms)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


class ERA5SurfaceForecastDataModule(LightningDataModule):
    def __init__(
        self,
        variable_data_paths: List = [
            '/mnt/weatherbench/temperature_2m.npy',
            '/mnt/weatherbench/wind_u_10m.npy',
            '/mnt/weatherbench/wind_v_10m.npy',
        ],
        predict_range: int = 6,
        train_val_test_split: Tuple[int, int, int] = (45000, 5000, 8439),
        normalize = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Normalize(mean=[277.0595, -0.05025468, 0.18755548], std=[21.289722, 5.5454874, 4.764006]) if normalize else None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = ERA5SurfaceForecast(self.hparams.variable_data_paths, self.hparams.predict_range, self.transforms)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

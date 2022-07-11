import glob
import os

import torch
import torchdata.datapipes as dp
from src.datamodules.era5_datapipe import (
    ERA5,
    ERA5Forecast,
    ERA5Npy,
    ERA5Zarr,
    IndividualDataIter,
    IndividualForecastDataIter,
)
from torch.utils.data import DataLoader

NPY = False
NPY_PATH = "/datadrive/datasets/1.40625deg_yearly_np/train"
ZARRY_PATH = "/datadrive/datasets/1.40625deg_yearly/train"


def collate_forecast_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    return inp, out


def collate_fn(batch):
    inp = torch.stack([batch[i] for i in range(len(batch))])
    return inp


if NPY:
    READER = ERA5Npy
    lister = dp.iter.FileLister(NPY_PATH)
else:
    READER = ERA5Zarr
    lister = dp.iter.IterableWrapper(glob.glob(os.path.join(ZARRY_PATH, "*.zarr")))

batchsize = 32

# test forecast dataloader
# dp = (
#     IndividualForecastDataIter(
#         ERA5Forecast(
#             READER(
#                 lister.shuffle().sharding_filter(),  # shuffle at the year level  # needed for num_workers > 1
#                 variables=["z", "r", "u", "v", "t", "t2m", "u10", "v10"],
#             )
#         ),
#     )
#     .shuffle(buffer_size=1000)  # shuffle at the individual data level
#     .batch(batchsize)
#     .in_batch_shuffle()  # shuffle within a batch, probably not necessary
#     .collate(collate_forecast_fn)
# )


# for x, y in DataLoader(dp, batch_size=None):
#     print(x.shape, y.shape)
#     break
#     # import pdb

#     # pdb.set_trace()

# test pretrain dataloader
import torchvision

dumb_normalize = torchvision.transforms.Normalize(torch.randn(17), torch.rand(17))
dp = (
    IndividualDataIter(
        ERA5(
            READER(
                lister.shuffle().sharding_filter(),  # shuffle at the year level  # needed for num_workers > 1
                variables=["z", "r", "u", "v", "t", "t2m", "u10", "v10"],
            )
        ),
        dumb_normalize,
    )
    .shuffle(buffer_size=1000)  # shuffle at the individual data level
    .batch(batchsize)
    .in_batch_shuffle()  # shuffle within a batch, probably not necessary
    .collate(collate_fn)
)


for x in DataLoader(dp, batch_size=None):
    print(x.shape)
    break
    # import pdb

    # pdb.set_trace()

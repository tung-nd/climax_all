from tkinter import Y

import torch
import torchdata.datapipes as dp
from src.datamodules.era5_datapipe import ERA5Forecast, ERA5Npy, IndividualDataIter

NPY_PATH = "/mnt/data/1.40625/_yearly_np"


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    return inp, out


batchsize = 32
dp = (
    IndividualDataIter(
        ERA5Forecast(
            ERA5Npy(dp.iter.FileLister(NPY_PATH), variables=["t", "u10", "v10"],)
            .shuffle(buffer_size=4)
            .sharding_filter()
        ),
    )
    .shuffle(buffer_size=1000)
    .batch(batchsize)
    .in_batch_shuffle()
    .collate(collate_fn)
)

for x, y in dp:
    print(x.shape, y.shape)
    import pdb

    pdb.set_trace()


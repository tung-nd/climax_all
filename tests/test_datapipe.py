import torchdata.datapipes as dp
from src.datamodules.era5_datapipe import ERA5Forecast, ERA5Npy

dp = ERA5Forecast(
    ERA5Npy(
        dp.iter.FileLister("/mnt/data/1.40625/_yearly_np"),
        variables=["t", "u10", "v10"],
    ).shuffle(buffer_size=5)
)

for x, y in dp:
    print(x.shape, y.shape)
    import pdb
    pdb.set_trace()


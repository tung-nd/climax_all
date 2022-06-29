import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class ERA5Surface(Dataset):
    def __init__(self, paths, transforms=None):
        super(ERA5Surface, self).__init__()
        self.transforms = transforms
        self.np_mms = []
        for path in paths:
            print ('Loading numpy data from ' + path)
            self.np_mms.append(np.memmap(path, dtype='float32', mode='r', shape=(350640, 128, 256)))

    def __getitem__(self, index):
        np_data = [mm[index] for mm in self.np_mms]
        np_data = np.stack(np_data, axis=0)
        torch_data = torch.from_numpy(np_data)
        if self.transforms:
            torch_data = self.transforms(torch_data)
        return torch_data

    def __len__(self):
        return self.np_mms[0].shape[0]

# data_paths = [
#     '/mnt/weatherbench/temperature_2m',
#     '/mnt/weatherbench/wind_u_10m',
#     '/mnt/weatherbench/wind_v_10m',
# ]
# for data_path in data_paths:
#     print ('Saving numpy data from ' + data_path)
#     xr_dataset = xr.open_mfdataset(os.path.join(data_path, '*.nc'), combine='by_coords')
#     xr_data = xr_dataset[list(xr_dataset)[0]].astype(np.float32)
#     np_data = xr_data.to_numpy()
#     np_path = os.path.join('/mnt/weatherbench', data_path.split('/')[-1] + '.npy')
#     fp = np.memmap(np_path, dtype='float32', mode='w+', shape=(350640, 128, 256))
#     fp[:] = np_data[:]
#     del np_data
#     fp.flush()

# data_paths = [
#     '/mnt/weatherbench/temperature_2m.npy',
#     '/mnt/weatherbench/wind_u_10m.npy',
#     '/mnt/weatherbench/wind_v_10m.npy',
# ]
# dataset = ERA5Surface(data_paths)
# print (len(dataset))
# samples = dataset[:50]
# print (samples.shape)

import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


def create_era5_surface(data_dir_paths):
    """
    data_dir_paths
    create .npy data file from directories of netcdf files
    """
    for data_path in data_dir_paths:
        print ('Saving numpy data from ' + data_path)
        xr_dataset = xr.open_mfdataset(os.path.join(data_path, '*.nc'), combine='by_coords')
        xr_data = xr_dataset[list(xr_dataset)[0]].astype(np.float32)
        np_data = xr_data.to_numpy()
        np_path = os.path.join('/mnt/weatherbench', data_path.split('/')[-1] + '.npy')
        fp = np.memmap(np_path, dtype='float32', mode='w+', shape=(350640, 128, 256))
        fp[:] = np_data[:]
        del np_data
        fp.flush()


class ERA5Surface(Dataset):
    def __init__(self, paths, transforms=None):
        """
        paths: paths to npy files, each file is one climate variable
        transforms: data transformation
        """
        super(ERA5Surface, self).__init__()
        self.transforms = transforms
        self.data_mms = []
        for path in paths:
            self.data_mms.append(np.memmap(path, dtype='float32', mode='r', shape=(350640, 128, 256)))

    def __getitem__(self, index):
        np_data = [mm[index] for mm in self.data_mms]
        np_data = np.stack(np_data, axis=0)
        torch_data = torch.from_numpy(np_data)
        if self.transforms:
            torch_data = self.transforms(torch_data)
        return torch_data

    def __len__(self):
        return self.data_mms[0].shape[0]


class ERA5SurfaceForecast(Dataset):
    def __init__(self, paths, predict_range=6, transforms=None):
        """
        paths: paths to npy files, each file is one climate variable
        predict_range: how many hours we predict into the future
        transforms: data transformation
        """
        super(ERA5SurfaceForecast, self).__init__()
        self.transforms = transforms
        self.input_mms = []
        self.output_mms = []
        for path in paths:
            all_data = np.memmap(path, dtype='float32', mode='r', shape=(350640, 128, 256))
            self.input_mms.append(all_data[0:-predict_range:predict_range])
            self.output_mms.append(all_data[predict_range::predict_range])

    def __getitem__(self, index):
        inp = [mm[index] for mm in self.input_mms]
        inp = np.stack(inp, axis=0)
        inp = torch.from_numpy(inp)

        out = [mm[index] for mm in self.output_mms]
        out = np.stack(out, axis=0)
        out = torch.from_numpy(out)

        if self.transforms:
            inp = self.transforms(inp)
            out = self.transforms(out)
        
        return inp, out

    def __len__(self):
        return self.input_mms[0].shape[0]


# data_paths = [
#     '/mnt/weatherbench/temperature_2m',
#     '/mnt/weatherbench/wind_u_10m',
#     '/mnt/weatherbench/wind_v_10m',
# ]
# create_era5_surface(data_paths)

# data_paths = [
#     '/mnt/weatherbench/temperature_2m.npy',
#     '/mnt/weatherbench/wind_u_10m.npy',
#     '/mnt/weatherbench/wind_v_10m.npy',
# ]
# dataset = ERA5Surface(data_paths)
# print (len(dataset))
# samples = dataset[:50]
# print (samples.shape)

# dataset = ERA5SurfaceForecast(data_paths, predict_range=6)
# print (len(dataset))
# for i in range(100):
#     print (dataset.output_mms[0][i] == dataset.input_mms[0][i+1])

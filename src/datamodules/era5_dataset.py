import os

import numpy as np
import torch
import xarray as xr
from src.datamodules import normalize_mean, normalize_std
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def create_era5_surface(data_dir_paths, save_dir='/datadrive/datasets/1.40625deg'):
    """
    data_dir_paths
    create .npy data file from directories of netcdf files
    """
    for data_path in data_dir_paths:
        print ('Saving numpy data from ' + data_path)
        xr_dataset = xr.open_mfdataset(os.path.join(data_path, '*.nc'), combine='by_coords')
        xr_data = xr_dataset[list(xr_dataset)[0]].astype(np.float32)
        np_data = xr_data.to_numpy()

        variable_name = data_path.split('/')[-1]
        np_path = os.path.join(save_dir, variable_name + '.npy')
        fp = np.memmap(np_path, dtype='float32', mode='w+', shape=(350640, 128, 256))
        fp[:] = np_data[:]
        del np_data
        fp.flush()


def create_era5_pressure_level(paths_level, save_dir='/datadrive/datasets/1.40625deg'):
    for data_path in paths_level.keys():
        print ('Saving numpy data from ' + data_path)
        xr_dataset = xr.open_mfdataset(os.path.join(data_path, '*.nc'), combine='by_coords')
        xr_data = xr_dataset[list(xr_dataset)[0]].astype(np.float32)
        all_levels = paths_level[data_path]
        for level in all_levels:
            print (f'Level {level}')
            xr_data_level = xr_data.sel(level = level)
            np_data = xr_data_level.to_numpy()

            variable_name = data_path.split('/')[-1] + f'_{level}hPa'
            np_path = os.path.join(save_dir, variable_name + '.npy')
            fp = np.memmap(np_path, dtype='float32', mode='w+', shape=(333120, 128, 256))
            fp[:] = np_data[:]
            del np_data
            fp.flush()


def get_transforms(variables):
    mean = [normalize_mean[v] for v in variables]
    std = [normalize_std[v] for v in variables]
    return transforms.Normalize(mean=mean, std=std)


class ERA5(Dataset):
    def __init__(self, root, variables):
        """
        paths: paths to npy files, each file is one climate variable
        transforms: data transformation
        """
        super(ERA5, self).__init__()
        self.transforms = get_transforms(variables)
        self.data_mms = []
        for var in variables:
            path = os.path.join(root, var + '.npy')
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


class ERA5Max(Dataset):
    def __init__(self, root, variables):
        """
        paths: paths to npy files, each file is one climate variable
        transforms: data transformation
        """
        super(ERA5Max, self).__init__()
        self.transforms = get_transforms(variables)
        self.data_mms = []
        for var in variables:
            path = os.path.join(root, var + '.npy')
            self.data_mms.append(np.memmap(path, dtype='float32', mode='r', shape=(350640, 128, 256)))

    def __getitem__(self, index):
        np_data = [mm[index] for mm in self.data_mms]
        np_data = np.stack(np_data, axis=0)
        torch_data = torch.from_numpy(np_data)
        if self.transforms:
            torch_data = self.transforms(torch_data)
        return torch_data, torch.amax(torch_data, dim=[1,2])

    def __len__(self):
        return self.data_mms[0].shape[0]


class ERA5Forecast(Dataset):
    def __init__(self, root, variables, predict_range=6):
        """
        paths: paths to npy files, each file is one climate variable
        predict_range: how many hours we predict into the future
        transforms: data transformation
        """
        super(ERA5Forecast, self).__init__()
        self.transforms = get_transforms(variables)
        self.input_mms = []
        self.output_mms = []
        for var in variables:
            path = os.path.join(root, var + '.npy')
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
#     '/datadrive/datasets/1.40625deg/2m_temperature',
#     '/datadrive/datasets/1.40625deg/10m_u_component_of_wind',
#     '/datadrive/datasets/1.40625deg/10m_v_component_of_wind',
# ]
# create_era5_surface(data_paths, '/datadrive/datasets/1.40625deg')

# paths_level = {
#     '/mnt/data_write/1.40625deg/geopotential': [50, 500, 850, 1000],
#     '/mnt/data_write/1.40625deg/u_component_of_wind': [500, 850, 1000],
#     '/mnt/data_write/1.40625deg/v_component_of_wind': [500, 850, 1000],
#     '/mnt/data_write/1.40625deg/temperature': [500, 850],
#     '/mnt/data_write/1.40625deg/relative_humidity': [500, 850],
# }
# create_era5_pressure_level(paths_level, '/mnt/data_write/1.40625deg')

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

import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def load_x_y(root_dir, list_simu, out_var):
    x_all, y_all = [], []
    for simu in list_simu:
        x = xr.open_dataset(os.path.join(root_dir, 'inputs_' + simu + '.nc')).to_array()
        x = x.to_numpy() # C, N, H, W
        x = x.transpose(1, 0, 2, 3) # N, C, H, W
        x_all.append(x)

        y = xr.open_dataset(os.path.join(root_dir, 'outputs_' + simu + '.nc')).mean(dim='member')
        y = y.assign({"pr": y.pr * 86400, "pr90": y.pr90 * 86400})
        y = y[out_var].to_array().to_numpy() # 1, N, H, W
        y = y.transpose(1, 0, 2, 3) # N, 1, H, W
        y_all.append(y)
    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    temp = xr.open_dataset(os.path.join(root_dir, 'inputs_' + list_simu[0] + '.nc'))
    lat = np.array(temp['latitude'])
    lon = np.array(temp['longitude'])

    return x_all, y_all, lat, lon

def split_train_val(x, y, train_ratio=0.9):
    shuffled_ids = np.random.permutation(x.shape[0])
    train_len = int(train_ratio * x.shape[0])
    train_ids = shuffled_ids[:train_len]
    val_ids = shuffled_ids[train_len:]
    return x[train_ids], y[train_ids], x[val_ids], y[val_ids]

class ClimateBenchDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, variables, out_variables, partition='train'):
        super().__init__()
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.variables = variables
        self.out_variables = out_variables
        self.partition = partition
    
        if partition == 'train':
            self.inp_transform = self.get_normalize(self.x)
            self.out_transform = self.get_normalize(self.y)
        else:
            self.inp_transform = None
            self.out_transform = None

    def get_normalize(self, data):
        mean = np.mean(data, axis=(0, 2, 3))
        std = np.std(data, axis=(0, 2, 3))
        return transforms.Normalize(mean, std)

    def set_normalize(self, inp_normalize, out_normalize): # for val and test
        self.inp_transform = inp_normalize
        self.out_transform = out_normalize

    def set_region_info(self, region_info):
        self.region_info = region_info

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        inp = self.inp_transform(torch.from_numpy(self.x[index]))
        out = self.out_transform(torch.from_numpy(self.y[index]))
        # lead times = 0
        lead_times = torch.Tensor([0.0]).to(dtype=inp.dtype)
        return inp.unsqueeze(0), out, lead_times, self.variables, self.out_variables, self.region_info

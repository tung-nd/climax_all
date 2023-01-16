import argparse
import os
from glob import glob

import numpy as np
import torch
import xarray as xr
from torchvision.transforms import transforms

from utils.metrics import lat_weighted_acc, lat_weighted_rmse

VAR_MAP = {
    'z500': 'geopotential',
    't850': 'temperature',
    't2m': '2m_temperature',
    'u10': '10m_u_component_of_wind',
    'v10': '10m_v_component_of_wind'
}

TASK_TO_STEP = {
    '6hrs': '0 days 06:00:00',
    '1day': '1 days',
    '3days': '3 days',
    '5days': '5 days',
    '7days': '7 days',
    '2weeks': '14 days'
}

VAR_CODE = {
    'z500': 'z_500',
    't850': 't_850',
    't2m': 't2m',
    'u10': 'u10',
    'v10': 'v10'
}

LOG_DAYS = {
    '6hrs': 0,
    '1day': 1,
    '3days': 3,
    '5days': 5,
    '7days': 7,
    '2weeks': 14
}

def eval_ifs(var, task, root_dir):
    gt_dir = os.path.join(root_dir, '5.625deg', VAR_MAP[var])
    ifs_dir = os.path.join(root_dir, 'TIGGE_5.625deg', var)
    gt_paths, ifs_paths = [], []
    clim_path = os.path.join(root_dir, f"era5_forecast_5.625deg_{task}/test/climatology.npz")
    clim = np.load(clim_path)
    clim = clim[VAR_CODE[var]]
    clim = torch.from_numpy(clim)
    lat = np.load(os.path.join(root_dir, f"era5_forecast_5.625deg_{task}/lat.npy"))

    for year in [2018]:
        gt_paths.extend(glob(os.path.join(gt_dir, f"*{year}*.nc")))
        ifs_paths.extend(glob(os.path.join(ifs_dir, f"*{year}*.nc")))

    gt = xr.open_mfdataset(gt_paths, combine="by_coords")
    ifs = xr.open_mfdataset(ifs_paths, combine="by_coords")
    ifs = ifs.sel(step=TASK_TO_STEP[task])

    time_ids = np.isin(gt.time, ifs.valid_time)
    gt = gt.sel(time=time_ids)

    if var == 'z500':
        gt = gt.sel(level=500)
    if var == 't850':
        gt = gt.sel(level=850)

    gt_np = gt.to_array().to_numpy().squeeze()
    ifs_np = ifs.to_array().to_numpy().squeeze()[:len(gt_np)]

    gt_tensor = torch.from_numpy(gt_np).unsqueeze(1).unsqueeze(2)
    ifs_tensor = torch.from_numpy(ifs_np).unsqueeze(1).unsqueeze(2)
    if var == 'z500':
        ifs_tensor = ifs_tensor * 9.80665 # convert geopotential height to geopotential
    
    nan_ids = torch.isnan(ifs_tensor) | torch.isinf(ifs_tensor)
    gt_tensor = gt_tensor[~nan_ids].reshape(-1, 1, 1, 32, 64)
    ifs_tensor = ifs_tensor[~nan_ids].reshape(-1, 1, 1, 32, 64)

    transform = transforms.Normalize(np.array([0.0]), np.array([1.0]))

    loss_dict_rmse = lat_weighted_rmse(ifs_tensor, gt_tensor, transform, [var], lat, [1], [LOG_DAYS[task]], clim)
    loss_dict_acc = lat_weighted_acc(ifs_tensor, gt_tensor, transform, [var], lat, [1], [LOG_DAYS[task]], clim)

    print (loss_dict_rmse)
    print (loss_dict_acc)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--var", type=str, required=True, choices=['z500', 't850', 't2m', 'u10', 'v10'])
    parser.add_argument("--task", type=str, required=True, choices=['6hrs', '1day', '3days', '5days', '7days', '2weeks'])
    parser.add_argument("--root_dir", type=str, default='/datadrive/datasets')

    args = parser.parse_args()

    eval_ifs(args.var, args.task, args.root_dir)


if __name__ == "__main__":
    main()
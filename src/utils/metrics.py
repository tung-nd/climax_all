import numpy as np
import torch
from pytorch_pfn_extras import to


def mse(pred, y, vars):
    """
    y: [N, 3, H, W]
    pred: [N, L, p*p*3]
    vars: list of variable names
    """
    loss = (pred - y) ** 2
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[var] = torch.mean(loss[:, i])
    loss_dict["loss"] = torch.mean(torch.sum(loss, dim=1))

    return loss_dict


def lat_weighted_rmse(pred, y, vars):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    """
    error = (pred - y) ** 2  # [N, T, 3, H, W]

    # lattitude weights
    lat = np.load("lat.npy")
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(error.device)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            for step in range(y.shape[1]):
                loss_dict[f"w_rmse_{var}_step_{step+1}"] = torch.sqrt(torch.mean(error[:, step, i] * w_lat))

    loss_dict["w_rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])
    return loss_dict


def lat_weighted_acc(pred, y, vars):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    TODO: subtract the climatology
    """
    # lattitude weights
    lat = np.load("lat.npy")
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(pred.device)  # [1, H, 1]

    # clim = torch.mean(y, dim=1, keepdim=True)
    # pred = pred - clim
    # y = y - clim
    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            for step in range(y.shape[1]):
                pred_prime = pred[:, step, i] - torch.mean(pred[:, step, i])
                y_prime = y[:, step, i] - torch.mean(y[:, step, i])
                loss_dict[f"acc_{var}_step_{step+1}"] = torch.sum(w_lat * pred_prime * y_prime) / torch.sqrt(
                    torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
                )

    loss_dict["acc"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])
    return loss_dict


# def compute_weighted_acc(da_fc, da_true, mean_dims):
#     """
#     Compute the ACC with latitude weighting from two xr.DataArrays.
#     WARNING: Does not work if datasets contain NaNs
#     Args:
#         da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
#         da_true (xr.DataArray): Truth.
#         mean_dims: dimensions over which to average score
#     Returns:
#         acc: Latitude weighted acc
#     """

#     clim = da_true.mean("time")
#     try:
#         t = np.intersect1d(da_fc.time, da_true.time)
#         fa = da_fc.sel(time=t) - clim
#     except AttributeError:
#         t = da_true.time.values
#         fa = da_fc - clim
#     a = da_true.sel(time=t) - clim

#     weights_lat = np.cos(np.deg2rad(da_fc.lat))
#     weights_lat /= weights_lat.mean()
#     w = weights_lat

#     fa_prime = fa - fa.mean()
#     a_prime = a - a.mean()

#     acc = np.sum(w * fa_prime * a_prime) / np.sqrt(
#         np.sum(w * fa_prime ** 2) * np.sum(w * a_prime ** 2)
#     )
#     return acc


# pred = torch.randn(2, 4, 3, 128, 256).cuda()
# y = torch.randn(2, 4, 3, 128, 256).cuda()
# vars = ["x", "y", "z"]
# print(lat_weighted_rmse(pred, y, vars))
# print(lat_weighted_acc(pred, y, vars))

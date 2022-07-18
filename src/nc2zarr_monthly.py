import glob
import os

import click
import numpy as np
import xarray as xr
import zarr.storage
from tqdm import tqdm

from datamodules import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR

zarr.storage.default_compressor = None

MONTHS = range(1, 12 + 1)


def nc2zarr(path, variables, years, save_dir, partition):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)
    if partition == "train":
        normalize_mean = {}
        normalize_std = {}
    for year in tqdm(years):
        monthly_dataset = {k: None for k in MONTHS}
        for var in variables:
            ps = glob.glob(os.path.join(path, var, f"*{year}*.nc"))
            ds = xr.open_mfdataset(
                ps, combine="by_coords", parallel=True
            )  # dataset for a single variable
            code = NAME_TO_VAR[var]

            if len(ds[code].shape) == 3:  # surface level variables
                ds[code] = ds[code].expand_dims("val", axis=1)
            else:  # multiple-level variables, only use a subset
                assert len(ds[code].shape) == 4
                ds = ds.sel(level=DEFAULT_PRESSURE_LEVELS[code])
                ds = ds.rename({"level": "level_" + code})

            if partition == "train":  # compute mean and std of each var in each year
                var_mean_yearly = ds[code].mean(axis=(0, 2, 3)).to_numpy()
                var_std_yearly = ds[code].std(axis=(0, 2, 3)).to_numpy()
                if var not in normalize_mean:
                    normalize_mean[var] = [var_mean_yearly]
                    normalize_std[var] = [var_std_yearly]
                else:
                    normalize_mean[var].append(var_mean_yearly)
                    normalize_std[var].append(var_std_yearly)

            ds_group_by_month = ds.groupby("time.month")
            for month in MONTHS:  # sharding monthly
                if monthly_dataset[month] is None:
                    monthly_dataset[month] = ds_group_by_month[month]
                else:
                    monthly_dataset[month] = monthly_dataset[month].merge(
                        ds_group_by_month[month]
                    )

        for month in MONTHS:
            monthly_dataset[month].to_zarr(
                os.path.join(save_dir, partition, f"{year}_{month}.zarr"), mode="w",
            )

    if partition == "train":
        for var in normalize_mean.keys():
            normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
            normalize_std[var] = np.stack(normalize_std[var], axis=0)

        for var in normalize_mean.keys():  # aggregate over the years
            mean, std = normalize_mean[var], normalize_std[var]
            # var(X) = E[var(X|Y)] + var(E[X|Y])
            variance = (
                (std ** 2).mean(axis=0)
                + (mean ** 2).mean(axis=0)
                - mean.mean(axis=0) ** 2
            )
            std = np.sqrt(variance)
            # E[X] = E[E[X|Y]]
            mean = mean.mean(axis=0)
            normalize_mean[var] = mean
            normalize_std[var] = std

        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--variables",
    "-v",
    type=click.STRING,
    multiple=True,
    default=[
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "u_component_of_wind",
        "v_component_of_wind",
        "geopotential",
        "temperature",
        "relative_humidity",
    ],
)
@click.option("--start_train_year", type=int, default=1979)
@click.option("--start_val_year", type=int, default=2013)
@click.option("--start_test_year", type=int, default=2015)
@click.option("--end_year", type=int, default=2017)
def main(path, variables, start_train_year, start_val_year, start_test_year, end_year):
    assert (
        start_val_year > start_train_year
        and start_test_year > start_val_year
        and end_year > start_test_year
    )
    train_years = range(start_train_year, start_val_year)
    val_years = range(start_val_year, start_test_year)
    test_years = range(start_test_year, end_year)

    if len(variables) <= 3:  # small dataset for testing new models
        yearly_datapath = os.path.join(
            os.path.dirname(path), f"{os.path.basename(path)}_yearly_small"
        )
    else:
        yearly_datapath = os.path.join(
            os.path.dirname(path), f"{os.path.basename(path)}_yearly"
        )
    os.makedirs(yearly_datapath, exist_ok=True)

    nc2zarr(path, variables, train_years, yearly_datapath, "train")
    nc2zarr(path, variables, val_years, yearly_datapath, "val")
    nc2zarr(path, variables, test_years, yearly_datapath, "test")


if __name__ == "__main__":
    main()

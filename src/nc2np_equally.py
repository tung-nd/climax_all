import glob
import os

import click
import numpy as np
import xarray as xr
from tqdm import tqdm

from datamodules import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR

HOURS_PER_YEAR = 8760  # 365-day year


def nc2np(path, variables, years, save_dir, partition, num_shards_per_year):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)
    if partition == "train":
        normalize_mean = {}
        normalize_std = {}
    for year in tqdm(years):
        np_vars = {}
        for var in variables:
            ps = glob.glob(os.path.join(path, var, f"*{year}*.nc"))
            ds = xr.open_mfdataset(ps, combine="by_coords", parallel=True)  # dataset for a single variable
            code = NAME_TO_VAR[var]

            if len(ds[code].shape) == 3:  # surface level variables
                ds[code] = ds[code].expand_dims("val", axis=1)
            else:  # multiple-level variables, only use a subset
                assert len(ds[code].shape) == 4
                ds = ds.sel(level=DEFAULT_PRESSURE_LEVELS[code])

            # remove the last 24 hours if this year has 366 days
            np_vars[code] = ds[code].to_numpy()[:HOURS_PER_YEAR]

            if partition == "train":  # compute mean and std of each var in each year
                var_mean_yearly = np_vars[code].mean(axis=(0, 2, 3))
                var_std_yearly = np_vars[code].std(axis=(0, 2, 3))
                if var not in normalize_mean:
                    normalize_mean[var] = [var_mean_yearly]
                    normalize_std[var] = [var_std_yearly]
                else:
                    normalize_mean[var].append(var_mean_yearly)
                    normalize_std[var].append(var_std_yearly)

        assert HOURS_PER_YEAR % num_shards_per_year == 0
        num_hrs_per_shard = HOURS_PER_YEAR // num_shards_per_year
        for shard_id in range(num_shards_per_year):
            start_id = shard_id * num_hrs_per_shard
            end_id = start_id + num_hrs_per_shard
            sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
            np.savez(
                os.path.join(save_dir, partition, f"{year}_{shard_id}.npz"),
                **sharded_data,
            )

    if partition == "train":
        for var in normalize_mean.keys():
            normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
            normalize_std[var] = np.stack(normalize_std[var], axis=0)

        for var in normalize_mean.keys():  # aggregate over the years
            mean, std = normalize_mean[var], normalize_std[var]
            # var(X) = E[var(X|Y)] + var(E[X|Y])
            variance = (std**2).mean(axis=0) + (mean**2).mean(axis=0) - mean.mean(axis=0) ** 2
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
@click.option("--num_shards", type=int, default=10)
def main(
    path,
    variables,
    start_train_year,
    start_val_year,
    start_test_year,
    end_year,
    num_shards,
):
    assert start_val_year > start_train_year and start_test_year > start_val_year and end_year > start_test_year
    train_years = range(start_train_year, start_val_year)
    val_years = range(start_val_year, start_test_year)
    test_years = range(start_test_year, end_year)

    if len(variables) <= 3:  # small dataset for testing new models
        yearly_datapath = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}_equally_small_np")
    else:
        yearly_datapath = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}_equally_np")
    os.makedirs(yearly_datapath, exist_ok=True)

    nc2np(path, variables, train_years, yearly_datapath, "train", num_shards)
    nc2np(path, variables, val_years, yearly_datapath, "val", num_shards)
    nc2np(path, variables, test_years, yearly_datapath, "test", num_shards)


if __name__ == "__main__":
    main()

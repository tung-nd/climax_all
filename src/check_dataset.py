### Check if every file in the dataset is openable
import glob
import os

import click
import xarray as xr

from datamodules import NAME_TO_VAR


def get_shape(root_dir, var, year):
    try:
        ps = glob.glob(os.path.join(root_dir, var, f"*{year}*.nc"))
        ds = xr.open_mfdataset(ps, combine="by_coords", parallel=True)  # dataset for a single variable
        code = NAME_TO_VAR[var]
        return ds[code].shape
    except:
        return -1


def check_dataset(root_dir, years, var):
    print(f"Checking variable {var}")
    for year in years:
        shape = get_shape(root_dir, var, year)
        if shape == -1:  # opening file faield
            print(f"Error opening file year {year}, checking failed")
        else:
            print(f"Year {year}, shape {shape}")


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--variable", "-v", type=str)
@click.option("--start_year", type=int, default=1959)
@click.option("--end_year", type=int, default=2022)
def main(path, variable, start_year, end_year):
    assert end_year > start_year
    years = range(start_year, end_year)
    check_dataset(path, years, variable)


if __name__ == "__main__":
    main()

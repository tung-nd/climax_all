import glob
import os

import click
import numpy as np
import xarray as xr


def nc2np(path, variables, years):
    newdatapath = os.path.join(
        os.path.dirname(path), f"{os.path.basename(path)}_yearly_np"
    )
    os.makedirs(newdatapath, exist_ok=True)

    for year in years:
        paths = []
        for var in variables:
            ps = glob.glob(os.path.join(path, var, f"*{year}*.nc"))
            paths.extend(ps)

        ds = xr.open_mfdataset(paths, combine="by_coords", parallel=False)

        # TODO: check if this is necessary
        if "u10" in ds:
            ds["u10"] = ds.u10.expand_dims("val", axis=1)

        if "v10" in ds:
            ds["v10"] = ds.v10.expand_dims("val", axis=1)

        np_vars = {}
        for k in ds.keys():
            np_vars[k] = ds[k].to_numpy()

        np.savez(os.path.join(newdatapath, f"{year}.npz"), **np_vars)


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--variables",
    "-v",
    type=click.STRING,
    multiple=True,
    default=["temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"],
)
@click.option("--start_year", type=int, default=1979)
@click.option("--end_year", type=int, default=1980)
def main(path, variables, start_year, end_year):
    years = range(start_year, end_year)
    nc2np(str(path), variables, years)


if __name__ == "__main__":
    main()

import glob
import os

import click
import xarray as xr
import zarr.storage

zarr.storage.default_compressor = None


def nc2zarr(path, variables, years):
    newdatapath = os.path.join(
        os.path.dirname(path), f"{os.path.basename(path)}_yearly"
    )
    os.makedirs(newdatapath, exist_ok=True)

    for year in years:
        paths = []
        for var in variables:
            ps = glob.glob(os.path.join(path, var, f"*{year}*.nc"))
            paths.extend(ps)

        ds = xr.open_mfdataset(paths, combine="by_coords", parallel=False)

        if "u10" in ds:
            ds["u10"] = ds.u10.expand_dims("val", axis=1)

        if "v10" in ds:
            ds["v10"] = ds.v10.expand_dims("val", axis=1)

        ds.to_zarr(
            os.path.join(newdatapath, f"{year}.zarr"), mode="w",
        )


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
    nc2zarr(path, variables, years)


if __name__ == "__main__":
    main()

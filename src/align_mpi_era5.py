import glob
import os

import numpy as np
import xarray as xr
from tqdm import tqdm

mpi_dir = '/datadrive/datasets/CMIP6/MPI-ESM/5.625deg'
era5_dir = '/datadrive/datasets/1.40625deg'

new_mpi_dir = '/datadrive/datasets/CMIP6/MPI-ESM/5.625deg_aligned'
new_era5_dir = '/datadrive/datasets/1.40625deg_aligned'

mpi_res = '5.625deg'
era5_res = '1.40625deg'

all_vars = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_temperature',
    'geopotential',
    'specific_humidity',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind'
]

era5_years = list(range(1979, 2019))

def get_mpi_str(year):
    if year < 1980:
        str = '197501010600-198001010000'
    elif year >= 1980 and year < 1985:
        str = '198001010600-198501010000'
    elif year >= 1985 and year < 1990:
        str = '198501010600-199001010000'
    elif year >= 1990 and year < 1995:
        str = '199001010600-199501010000'
    elif year >= 1995 and year < 2000:
        str = '199501010600-200001010000'
    elif year >= 2000 and year < 2005:
        str = '200001010600-200501010000'
    elif year >= 2005 and year < 2010:
        str = '200501010600-201001010000'
    elif year >= 2010 and year < 2015:
        str = '201001010600-201501010000'
    else:
        str = None
    
    return str

for var in all_vars:
    print ('Aligning', var)
    os.makedirs(os.path.join(new_mpi_dir, var), exist_ok=True)
    os.makedirs(os.path.join(new_era5_dir, var), exist_ok=True)
    for year in tqdm(era5_years):
        mpi_str = get_mpi_str(year)
        if mpi_str is not None:
            mpi_ds = xr.open_dataset(os.path.join(mpi_dir, var, f"{var}_{mpi_str}_{mpi_res}.nc"))
            era5_ds = xr.open_dataset(os.path.join(era5_dir, var, f"{var}_{year}_{era5_res}.nc"))

            time_ids = np.isin(mpi_ds.time, era5_ds.time)
            mpi_ds = mpi_ds.sel(time=time_ids)

            time_ids = np.isin(era5_ds.time, mpi_ds.time)
            era5_ds = era5_ds.sel(time=time_ids)

            mpi_ds.to_netcdf(os.path.join(new_mpi_dir, var, f"{var}_{year}_{mpi_res}.nc"))
            era5_ds.to_netcdf(os.path.join(new_era5_dir, var, f"{var}_{year}_{era5_res}.nc"))
import sys
from os.path import abspath, dirname

sys.path.append(dirname(dirname(abspath(__file__))))

normalize_mean = {
    "2m_temperature": 277.0595,
    "10m_u_component_of_wind": -0.05025468,
    "10m_v_component_of_wind": 0.18755548,
}

normalize_std = {
    "2m_temperature": 21.289722,
    "10m_u_component_of_wind": 5.5454874,
    "10m_v_component_of_wind": 4.764006,
}

NAME_TO_VAR = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "geopotential": "z",
    "temperature": "t",
    "relative_humidity": "r",
    "total_precipitation": "tp",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

NAME_LEVEL_TO_VAR_LEVEL = {
    "geopotential_1000": "z_1000",
    "geopotential_850": "z_850",
    "geopotential_500": "z_500",
    "geopotential_50": "z_50",
    "relative_humidity_850": "r_850",
    "relative_humidity_500": "r_500",
    "u_component_of_wind_1000": "u_1000",
    "u_component_of_wind_850": "u_850",
    "u_component_of_wind_500": "u_500",
    "v_component_of_wind_1000": "v_1000",
    "v_component_of_wind_850": "v_850",
    "v_component_of_wind_500": "v_500",
    "temperature_850": "t_850",
    "temperature_500": "t_500",
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "total_precipitation": "tp",
}

VAR_LEVEL_TO_NAME_LEVEL = {v: k for k, v in NAME_LEVEL_TO_VAR_LEVEL.items()}

DEFAULT_PRESSURE_LEVELS = {
    "u": [1000, 850, 500],
    "v": [1000, 850, 500],
    "z": [1000, 850, 500, 50],
    "t": [850, 500],
    "r": [850, 500],
}

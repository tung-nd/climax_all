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

DEFAULT_PRESSURE_LEVELS = {
    "u": [1000, 850, 500],
    "v": [1000, 850, 500],
    "z": [1000, 850, 500, 50],
    "t": [850, 500],
    "r": [850, 500],
}

VAR_TO_NAME_LEVEL = {}
for v in VAR_TO_NAME.keys():
    var_name = VAR_TO_NAME[v]
    if v not in DEFAULT_PRESSURE_LEVELS: # v is a single level variable
        VAR_TO_NAME_LEVEL[v] = [var_name]
    else:
        VAR_TO_NAME_LEVEL[v] = []
        for p in DEFAULT_PRESSURE_LEVELS[v]:
            VAR_TO_NAME_LEVEL[v].append(f'{var_name}_{p}')
# {
#     't2m': ['2m_temperature'],
#     'u10': ['10m_u_component_of_wind'],
#     'v10': ['10m_v_component_of_wind'],
#     'msl': ['mean_sea_level_pressure'],
#     'sp': ['surface_pressure'],
#     'u': ['u_component_of_wind_1000', 'u_component_of_wind_850', 'u_component_of_wind_500'],
#     'v': ['v_component_of_wind_1000', 'v_component_of_wind_850', 'v_component_of_wind_500'],
#     'z': ['geopotential_1000', 'geopotential_850', 'geopotential_500', 'geopotential_50'],
#     't': ['temperature_850', 'temperature_500'],
#     'r': ['relative_humidity_850', 'relative_humidity_500'],
#     'tp': ['total_precipitation']
# }

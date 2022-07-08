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

NAME_MAP = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "geopotential": "z",
    "temperature": "t",
    "relative_humidity": "r",
}

DEFAULT_PRESSURE_LEVELS = {
    "u": [1000, 850, 500],
    "v": [1000, 850, 500],
    "z": [1000, 850, 500, 50],
    "t": [850, 500],
    "rh": [850, 500],
}

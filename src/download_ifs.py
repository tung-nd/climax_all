import argparse
import os

from ecmwfapi import ECMWFDataServer

PARAM_DICT = {
    'z500': '156',
    't850': '130',
    't2m': '167',
    'u10': '165',
    'v10': '166'
}

LEVTYPE = {
    'z500': 'pl',
    't850': 'pl',
    't2m': 'sfc',
    'u10': 'sfc',
    'v10': 'sfc'
}

LEVELIST = {
    'z500': '500',
    't850': '850',
    't2m': None,
    'u10': None,
    'v10': None
}

NUM_DAYS = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31
}

def download_tigge(var, year, month, save_dir):
    save_dir = os.path.join(save_dir, var)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    

    n_days = NUM_DAYS[month]
    date = f"{year}-{month:02}-01/to/{year}-{month:02}-{n_days}"

    request = {
        "class": "ti",
        "dataset": "tigge",
        "date": date,
        "expver": "prod",
        "grid": "0.5/0.5",
        "levtype": LEVTYPE[var],
        "origin": "ecmf",
        "param": PARAM_DICT[var],
        "step": "6/24/72/120/168/336",
        "time": "12:00:00",
        "type": "cf",
        "target": os.path.join(save_dir, f"{year}-{month:02}.grib"),
    }
    if var in ['z500', 't850']:
        request['levelist'] = str(LEVELIST[var])

    print ('Download request: ', request)
    server = ECMWFDataServer()
    server.retrieve(request)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--var", type=str, required=True, choices=['z500', 't850', 't2m', 'u10', 'v10'])
    parser.add_argument("--year", type=int, required=True, choices=[2017, 2018])
    parser.add_argument("--month", type=int, required=True, choices=list(range(1, 13)))
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    download_tigge(args.var, args.year, args.month, args.save_dir)


if __name__ == "__main__":
    main()
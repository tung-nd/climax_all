import argparse
import os

import cdsapi

latlong_map = {'us': [50, -125, 24, -67], 'global': 'global'}


def download_era5(args):
    years = [str(i) for i in range(args.start_year, args.end_year+1)]
    months = [
        "01", "02", "03", "04",
        "05", "06", "07", "08",
        "09", "10", "11", "12",
    ]
    days = [
        '01', '02', '03',
        '04', '05', '06',
        '07', '08', '09',
        '10', '11', '12',
        '13', '14', '15',
        '16', '17', '18',
        '19', '20', '21',
        '22', '23', '24',
        '25', '26', '27',
        '28', '29', '30',
        '31',
    ]
    times = [
        '00:00', '01:00', '02:00',
        '03:00', '04:00', '05:00',
        '06:00', '07:00', '08:00',
        '09:00', '10:00', '11:00',
        '12:00', '13:00', '14:00',
        '15:00', '16:00', '17:00',
        '18:00', '19:00', '20:00',
        '21:00', '22:00', '23:00',
    ]
    area = latlong_map[args.area]
    c = cdsapi.Client()
    if args.type == 'surface':
        for y in years:
            print(f"Downloading surface data for year {y}")
            fn = os.path.join(args.root, f"era5_surface_{y}.nc")
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": [
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "2m_temperature",
                        "mean_sea_level_pressure",
                        "surface_pressure",
                    ],
                    "year": y,
                    "month": months,
                    "day": days,
                    "time": times,
                    "area": area,
                },
                fn,
            )
            print("=" * 100)
    elif args.type == 'pressure':
        pressure_levels = args.pressures
        for p in pressure_levels:
            for y in years:
                print(f"Downloading data for pressure {p} year {y}")
                fn = os.path.join(args.root, f"era5_surface_{y}.nc")
                c.retrieve(
                    "reanalysis-era5-pressure-levels",
                    {
                        "product_type": "reanalysis",
                        "format": "netcdf",
                        "variable": [
                            "geopotential",
                            "relative_humidity",
                            "temperature",
                            "u_component_of_wind",
                            "v_component_of_wind",
                        ],
                        "pressure_level": p,
                        "year": y,
                        "month": months,
                        "day": days,
                        "time": times,
                        "area": area,
                    },
                    fn,
                )
                print("=" * 100)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="/mnt/ear5")
    parser.add_argument("--type", type=str, required=True, choices=["pressure", "surface"])
    parser.add_argument("--area", type=str, default='global')

    parser.add_argument("--start_year", type=int, default=1959)
    parser.add_argument("--end_year", type=int, default=2022)

    parser.add_argument("--pressures", nargs="+", default=["50", "500", "850", "1000"])

    args = parser.parse_args()

    download_era5(args)


if __name__ == "__main__":
    main()

import argparse
import os
import subprocess


def download_weatherbench(args):
    resolution = args.resolution
    variable = args.variable
    url = (
        "https://dataserv.ub.tum.de/s/m1524895"
        "/download?path=%2F{resolution}deg%2F{variable}&files={variable}_{resolution}deg.zip"
    ).format(resolution=resolution, variable=variable)
    cmd = ["wget", "--no-check-certificate", f'"{url}"', "-O", os.path.join(args.root, variable + ".zip")]
    # print (" ".join(cmd))
    subprocess.run(["wget", "--no-check-certificate", url, "-O", os.path.join(args.root, variable + ".zip")])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="/datadrive/datasets/1.40625deg")
    parser.add_argument("--variable", type=str, required=True)
    parser.add_argument("--resolution", type=str, default="1.40625")

    args = parser.parse_args()

    download_weatherbench(args)


if __name__ == "__main__":
    main()

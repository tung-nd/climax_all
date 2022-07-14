import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.utilities.cli import LightningCLI
from torchvision.transforms import transforms

from src.datamodules.era5_datapipe_module import ERA5DataPipeModule
from src.models.mae_module import MAELitModule

inv_normalize = transforms.Normalize(
    mean=[-277.0595 / 21.289722, 0.05025468 / 5.5454874, -0.18755548 / 4.764006],
    std=[1 / 21.289722, 1 / 5.5454874, 1 / 4.764006],
)  # 2m_temperature, 10m_u_component_of_wind, 10m_v_component_of_wind


def save_img(data, masked_data, pred, save_dir, filename):
    vmin_temp, vmax_temp = data[0].min(), data[0].max()
    vmin_wind_u, vmax_wind_u = data[1].min(), data[1].max()
    vmin_wind_v, vmax_wind_v = data[2].min(), data[2].max()

    fig, axes = plt.subplots(3, 4, figsize=(40, 12))
    cmap = plt.cm.get_cmap("RdBu").copy()
    cmap.set_under("lightgrey")

    # ground-truth
    im1 = axes[0][0].imshow(data[0], vmin=vmin_temp, vmax=vmax_temp)
    im1.set_cmap(cmap=cmap)
    fig.colorbar(im1, ax=axes[0][0])
    axes[0][0].set_title("gt_temperature")

    im2 = axes[1][0].imshow(data[1], vmin=vmin_wind_u, vmax=vmax_wind_u)
    im2.set_cmap(cmap=cmap)
    fig.colorbar(im2, ax=axes[1][0])
    axes[1][0].set_title("gt_wind_u")

    im3 = axes[2][0].imshow(data[2], vmin=vmin_wind_v, vmax=vmax_wind_v)
    im3.set_cmap(cmap=cmap)
    fig.colorbar(im3, ax=axes[2][0])
    axes[2][0].set_title("gt_wind_v")
    # -------------------------------------------------------------------

    # masked
    im4 = axes[0][1].imshow(masked_data[0], vmin=vmin_temp, vmax=vmax_temp)
    im4.set_cmap(cmap=cmap)
    fig.colorbar(im4, ax=axes[0][1])
    axes[0][1].set_title("masked_temperature")

    im5 = axes[1][1].imshow(masked_data[1], vmin=vmin_wind_u, vmax=vmax_wind_u)
    im5.set_cmap(cmap=cmap)
    fig.colorbar(im5, ax=axes[1][1])
    axes[1][1].set_title("masked_wind_u")

    im6 = axes[2][1].imshow(masked_data[2], vmin=vmin_wind_v, vmax=vmax_wind_v)
    im6.set_cmap(cmap=cmap)
    fig.colorbar(im6, ax=axes[2][1])
    axes[2][1].set_title("masked_wind_v")
    # -------------------------------------------------------------------

    # predictions
    im7 = axes[0][2].imshow(pred[0], vmin=vmin_temp, vmax=vmax_temp)
    im7.set_cmap(cmap=cmap)
    fig.colorbar(im7, ax=axes[0][2])
    axes[0][2].set_title("pred_temperature")

    im8 = axes[1][2].imshow(pred[1], vmin=vmin_wind_u, vmax=vmax_wind_u)
    im8.set_cmap(cmap=cmap)
    fig.colorbar(im8, ax=axes[1][2])
    axes[1][2].set_title("pred_wind_u")

    im9 = axes[2][2].imshow(pred[2], vmin=vmin_wind_v, vmax=vmax_wind_v)
    im9.set_cmap(cmap=cmap)
    fig.colorbar(im9, ax=axes[2][2])
    axes[2][2].set_title("pred_wind_v")
    # -------------------------------------------------------------------

    # difference
    im10 = axes[0][3].imshow(pred[0] - data[0])
    im10.set_cmap(cmap=cmap)
    fig.colorbar(im10, ax=axes[0][3])
    axes[0][3].set_title("diff_temperature")

    im11 = axes[1][3].imshow(pred[1] - data[1])
    im11.set_cmap(cmap=cmap)
    fig.colorbar(im11, ax=axes[1][3])
    axes[1][3].set_title("diff_wind_u")

    im12 = axes[2][3].imshow(pred[2] - data[2])
    im12.set_cmap(cmap=cmap)
    fig.colorbar(im12, ax=axes[2][3])
    axes[2][3].set_title("diff_wind_v")
    # -------------------------------------------------------------------

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, filename))


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--ckpt", type=str, required=True)
        parser.add_argument(
            "--save_dir",
            type=str,
            default="/home/t-tungnguyen/climate_pretraining/visualization_videomae",
        )
        parser.add_argument("--filename", type=str, default="model.png")


def main(model, dataset, args):
    os.makedirs(args.save_dir, exist_ok=True)

    dataset.setup()
    dataset = dataset.data_test

    data_sample = next(iter(dataset))[0]  # 8, 3, 128, 256

    # ground-truth
    ground_truth = inv_normalize(data_sample)

    # prediction
    pred, mask = model.forward(data_sample.unsqueeze(0))  # 8, 3, 128, 256

    mask = mask.squeeze().bool()
    masked_data = ground_truth.clone()
    masked_data[mask] = -1000.0

    pred = pred.squeeze()
    pred = inv_normalize(pred)

    for i in range(pred.shape[0]):
        save_img(
            ground_truth[i],
            masked_data[i],
            pred[i],
            args.save_dir,
            str(i) + "_" + args.filename,
        )


if __name__ == "__main__":
    cli = MyLightningCLI(
        model_class=MAELitModule,
        datamodule_class=ERA5DataPipeModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    state_dict = torch.load(cli.config.ckpt)["state_dict"]
    msg = cli.model.load_state_dict(state_dict)
    print(msg)

    main(cli.model, cli.datamodule, cli.config)

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.utilities.cli import LightningCLI
from torchvision.transforms import transforms

from src.datamodules.era5_datamodule import ERA5DataModule
from src.models.mae_module import MAELitModule


def get_data_traj(dataset, start_idx, steps):
    # visualize a trajectory from the dataset, starting from start_idx
    dataset_traj = []
    inv_normalize = transforms.Normalize(
        mean=[-277.0595/21.289722, 0.05025468/5.5454874, -0.18755548/4.764006],
        std=[1/21.289722, 1/5.5454874, 1/4.764006]
    )
    for i in range(steps):
        idx = start_idx + i
        inp, _ = dataset[idx]
        dataset_traj.append(inv_normalize(inp).numpy())
    dataset_traj = np.array(dataset_traj)
    return dataset_traj

def get_model_traj(model, dataset, start_idx, steps):
    inv_normalize = transforms.Normalize(
        mean=[-277.0595/21.289722, 0.05025468/5.5454874, -0.18755548/4.764006],
        std=[1/21.289722, 1/5.5454874, 1/4.764006]
    )
    x = dataset[start_idx][0].unsqueeze(0)
    pred_traj = [inv_normalize(x.squeeze()).numpy()]
    for _ in range(steps):
        x = model.forward(x)
        pred_traj.append(inv_normalize(x.squeeze()).numpy())
    pred_traj = np.array(pred_traj)
    return pred_traj


def save_img(data, masked_data, pred, save_dir, filename):
    vmin_temp, vmax_temp = data[0].min(), data[0].max()
    vmin_wind_u, vmax_wind_u = data[1].min(), data[1].max()
    vmin_wind_v, vmax_wind_v = data[2].min(), data[2].max()

    fig, axes = plt.subplots(3,3, figsize=(30, 12))
    cmap = plt.cm.get_cmap("RdBu").copy()
    cmap.set_under('lightgrey')


    # ground-truth
    im1 = axes[0][0].imshow(data[0], vmin=vmin_temp, vmax=vmax_temp)
    im1.set_cmap(cmap=cmap)
    fig.colorbar(im1, ax=axes[0][0])
    axes[0][0].set_title('gt_temperature')

    im2 = axes[1][0].imshow(data[1], vmin=vmin_wind_u, vmax=vmax_wind_u)
    im2.set_cmap(cmap=cmap)
    fig.colorbar(im2, ax=axes[1][0])
    axes[1][0].set_title('gt_wind_u')

    im3 = axes[2][0].imshow(data[2], vmin=vmin_wind_v, vmax=vmax_wind_v)
    im3.set_cmap(cmap=cmap)
    fig.colorbar(im3, ax=axes[2][0])
    axes[2][0].set_title('gt_wind_v')
    # -------------------------------------------------------------------

    # masked
    im4 = axes[0][1].imshow(masked_data[0], vmin=vmin_temp, vmax=vmax_temp)
    im4.set_cmap(cmap=cmap)
    fig.colorbar(im4, ax=axes[0][1])
    axes[0][1].set_title('masked_temperature')

    im5 = axes[1][1].imshow(masked_data[1], vmin=vmin_wind_u, vmax=vmax_wind_u)
    im5.set_cmap(cmap=cmap)
    fig.colorbar(im5, ax=axes[1][1])
    axes[1][1].set_title('masked_wind_u')

    im6 = axes[2][1].imshow(masked_data[2], vmin=vmin_wind_v, vmax=vmax_wind_v)
    im6.set_cmap(cmap=cmap)
    fig.colorbar(im6, ax=axes[2][1])
    axes[2][1].set_title('masked_wind_v')
    # -------------------------------------------------------------------

    # predictions
    im7 = axes[0][2].imshow(pred[0], vmin=vmin_temp, vmax=vmax_temp)
    im7.set_cmap(cmap=cmap)
    fig.colorbar(im7, ax=axes[0][2])
    axes[0][2].set_title('pred_temperature')

    im8 = axes[1][2].imshow(pred[1], vmin=vmin_wind_u, vmax=vmax_wind_u)
    im8.set_cmap(cmap=cmap)
    fig.colorbar(im8, ax=axes[1][2])
    axes[1][2].set_title('pred_wind_u')

    im9 = axes[2][2].imshow(pred[2], vmin=vmin_wind_v, vmax=vmax_wind_v)
    im9.set_cmap(cmap=cmap)
    fig.colorbar(im9, ax=axes[2][2])
    axes[2][2].set_title('pred_wind_v')
    # -------------------------------------------------------------------

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, filename))


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--ckpt", type=str, required=True)
        parser.add_argument("--save_dir", type=str, default="/home/t-tungnguyen/climate_pretraining/visualization_mae")
        parser.add_argument("--filename", type=str, default="model.gif")


def main(model, dataset, args):
    os.makedirs(args.save_dir, exist_ok=True)

    dataset.setup()
    dataset = dataset.data_test

    rand_idx = np.random.randint(low=0, high=len(dataset)) # choose a random index from the dataset
    data_sample = dataset[rand_idx]

    inv_normalize = transforms.Normalize(
        mean=[-277.0595/21.289722, 0.05025468/5.5454874, -0.18755548/4.764006],
        std=[1/21.289722, 1/5.5454874, 1/4.764006]
    )

    # ground-truth
    ground_truth = inv_normalize(data_sample)

    # prediction
    pred, mask = model.forward(data_sample.unsqueeze(0))

    mask = mask.squeeze().bool()
    masked_data = ground_truth.clone()
    masked_data[mask] = -1000.0

    pred = pred.squeeze()
    pred = inv_normalize(pred)

    save_img(ground_truth, masked_data, pred, args.save_dir, 'viz.png')

if __name__ == "__main__":
    cli = MyLightningCLI(
        model_class=MAELitModule,
        datamodule_class=ERA5DataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    state_dict = torch.load(cli.config.ckpt)['state_dict']
    msg = cli.model.load_state_dict(state_dict)
    print (msg)

    main(cli.model, cli.datamodule, cli.config)

import os

from pytorch_lightning.utilities.cli import LightningCLI

from datamodules.era5_surface_datamodule import ERA5SurfaceMaxDataModule
from models.vit_module import ViTLitModule


def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=ViTLitModule,
        datamodule_class=ERA5SurfaceMaxDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()

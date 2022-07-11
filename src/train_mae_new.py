import os

from pytorch_lightning.utilities.cli import LightningCLI

from models.mae_module import MAELitModule
from src.datamodules.era5_datapipe_module import ERA5DataPipeModule


def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=MAELitModule,
        datamodule_class=ERA5DataPipeModule,
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

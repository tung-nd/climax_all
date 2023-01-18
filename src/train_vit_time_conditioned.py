import os

from pytorch_lightning.utilities.cli import LightningCLI

from models.vit_continuous_module import ViTContinuousLitModule
from src.datamodules.finetune_time_conditioned_module import (
    FinetuneTimeConditionedModule,
)


def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=ViTContinuousLitModule,
        datamodule_class=FinetuneTimeConditionedModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    cli.datamodule.set_patch_size(cli.model.get_patch_size())

    normalization = cli.datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(cli.datamodule.hparams.max_predict_range)
    cli.model.set_val_clim(None)
    cli.model.set_test_clim(None)

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # test the trained model
    # cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path='best')


if __name__ == "__main__":
    main()
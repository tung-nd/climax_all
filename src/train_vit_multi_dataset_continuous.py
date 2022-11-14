import os

from pytorch_lightning import Trainer
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.utilities.cli import LightningCLI

from models.vit_continuous_module import ViTContinuousLitModule
from src.datamodules.multi_source_iterdataset_continuous_module import \
    MultiSourceTrainDatasetModule


def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=ViTContinuousLitModule,
        datamodule_class=MultiSourceTrainDatasetModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )

    # trainer = Trainer(
    #     default_root_dir=cli.trainer.default_root_dir,
    #     precision=cli.trainer.precision,
    #     gpus=cli.trainer.gpus,
    #     devices=cli.trainer.devices,
    #     num_nodes=cli.trainer.num_nodes,
    #     accelerator=cli.trainer.accelerator,
    #     min_epochs=cli.trainer.min_epochs,
    #     max_epochs=cli.trainer.max_epochs,
    #     enable_progress_bar=True,
    #     sync_batchnorm=True,
    #     enable_checkpointing=True,
    #     resume_from_checkpoint=None,
    #     limit_val_batches=cli.trainer.limit_val_batches,
    #     num_sanity_val_steps=cli.trainer.num_sanity_val_steps,
    #     fast_dev_run=cli.trainer.fast_dev_run,
    #     logger=cli.trainer.logger,
    #     callbacks=cli.trainer.callbacks,
    #     strategy=DeepSpeedStrategy(
    #         stage=1,
    #         # loss_scale=2**16,
    #         # min_loss_scale=2**10,
    #         # loss_scale_window=500,
    #         initial_scale_power=10,
    #         logging_batch_size_per_gpu=cli.datamodule.hparams.batch_size
    #     )
    # )

    trainer = cli.trainer

    os.makedirs(trainer.default_root_dir, exist_ok=True)

    # normalization = cli.datamodule.output_transforms
    # mean_norm, std_norm = normalization.mean, normalization.std
    # mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    # cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    # cli.model.set_pred_range(cli.datamodule.hparams.dict_predict_ranges)

    # fit() runs the training
    trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()

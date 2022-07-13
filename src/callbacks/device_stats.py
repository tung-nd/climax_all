from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT

# CPU device metrics
_CPU_VM_PERCENT = "cpu_vm_percent"
_CPU_PERCENT = "cpu_percent"
_CPU_SWAP_PERCENT = "cpu_swap_percent"


def get_cpu_stats():
    import psutil

    return {
        _CPU_VM_PERCENT: psutil.virtual_memory().percent,
        _CPU_PERCENT: psutil.cpu_percent(),
        _CPU_SWAP_PERCENT: psutil.swap_memory().percent,
    }


@CALLBACK_REGISTRY
class DeviceStatsMonitor2(Callback):
    r"""
    Automatically monitors and logs device stats during training stage. ``DeviceStatsMonitor``
    is a special callback as it requires a ``logger`` to passed as argument to the ``Trainer``.
    Raises:
        MisconfigurationException:
            If ``Trainer`` has no logger.
    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import DeviceStatsMonitor
        >>> device_stats = DeviceStatsMonitor() # doctest: +SKIP
        >>> trainer = Trainer(callbacks=[device_stats]) # doctest: +SKIP
    """

    def __init__(self, cpu_stats: Optional[bool] = None) -> None:
        self._cpu_stats = cpu_stats

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:

        if not trainer.loggers:
            raise MisconfigurationException(
                "Cannot use DeviceStatsMonitor callback with Trainer that has no logger."
            )

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if not trainer.loggers:
            raise MisconfigurationException(
                "Cannot use `DeviceStatsMonitor` callback with `Trainer(logger=False)`."
            )

        device = trainer.strategy.root_device
        device_stats = trainer.accelerator.get_device_stats(device)
        if self._cpu_stats and device.type != "cpu":
            # Don't query CPU stats twice if CPU is accelerator

            device_stats.update(get_cpu_stats())
        for logger in trainer.loggers:
            separator = logger.group_separator
            prefixed_device_stats = _prefix_metric_keys(
                device_stats, "on_train_batch_start", separator
            )
            logger.log_metrics(prefixed_device_stats, step=trainer.global_step)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if not trainer.loggers:
            raise MisconfigurationException(
                "Cannot use `DeviceStatsMonitor` callback with `Trainer(logger=False)`."
            )

        device = trainer.strategy.root_device
        device_stats = trainer.accelerator.get_device_stats(device)
        if self._cpu_stats and device.type != "cpu":
            # Don't query CPU stats twice if CPU is accelerator
            device_stats.update(get_cpu_stats())
        for logger in trainer.loggers:
            separator = logger.group_separator
            prefixed_device_stats = _prefix_metric_keys(
                device_stats, "on_train_batch_end", separator
            )
            logger.log_metrics(prefixed_device_stats, step=trainer.global_step)


def _prefix_metric_keys(
    metrics_dict: Dict[str, float], prefix: str, separator: str
) -> Dict[str, float]:
    return {prefix + separator + k: v for k, v in metrics_dict.items()}


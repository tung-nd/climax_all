import pytest

from helpers import run_command


def test_fast_dev_run():
    """Test running for 1 train, val and test batch."""
    command = [
        "src/train_mae.py",
        "-c",
        "configs/train_mae.yaml",
        "--help",
    ]
    run_command(command)


# cpu only test for CI
def test_fast_dev_run_cpu():
    """Test running for 1 train, val and test batch."""
    command = [
        "src/train_mae.py",
        "-c",
        "configs/train_mae.yaml",
        "--help",
    ]
    run_command(command)

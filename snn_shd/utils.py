import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from snn_shd import config


class MetricTracker:
    """Accumulates and averages scalar metrics across batches, weighted by batch size."""

    def __init__(self):
        self.totals: dict[str, float] = {}
        self.n_samples: int = 0

    def update(self, batch_size: int, **metrics: float) -> None:
        """
        Accumulates metrics of one batch.

        Args:
          batch_size: Number of samples in this batch.
          **metrics: Scalar values for this batch.
        """
        self.n_samples += batch_size
        for name, value in metrics.items():
            self.totals[name] = self.totals.get(name, 0.0) + value * batch_size

    def get_running_avg(self) -> dict[str, float]:
        """
        Returns:
          A dict mapping metric name to its running average.
        """
        return {name: total / self.n_samples for name, total in self.totals.items()}


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    Saves a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include ".pt" file extension.
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pt"), f"Name {model_name} should end with '.pt'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def get_data_paths(
    dir_path: str = config.DATA_DIR,
    train_filename: str = config.TRAIN_FILENAME,
    test_filename: str = config.TEST_FILENAME,
) -> tuple[Path, Path]:
    """
    Locates the SHD training and testing files inside a directory.

    Args:
      dir_path: Directory expected to contain files for training and test set.

    Returns:
      A tuple of (train_path, test_path).

    Raises:
      NotADirectoryError: If dir_path is not a directory.
      FileNotFoundError: If either expected file is missing.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"The path '{dir_path}' is not a directory.")
    train_path = dir_path / train_filename
    test_path = dir_path / test_filename

    if not train_path.is_file():
        raise FileNotFoundError(f"The file '{train_path}' does not exist.")
    if not test_path.is_file():
        raise FileNotFoundError(f"The file '{test_path}' does not exist.")
    return train_path, test_path


def create_writer(
    experiment_name: str, model_name: str, extra: str | None = None
) -> torch.utils.tensorboard.writer.SummaryWriter:
    """
    Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.
    """

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime(
        "%Y-%m-%d"
    )  # returns current date in YYYY-MM-DD format

    root = Path("runs")

    if extra:
        log_dir = root / timestamp / experiment_name / model_name / extra
    else:
        log_dir = root / timestamp / experiment_name / model_name

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


def write_scalar(writer, tag_scalar_dict, main_tag, epoch):
    writer.add_scalars(
        main_tag=main_tag,
        tag_scalar_dict=tag_scalar_dict,
        global_step=epoch,
    )


def write_results(
    writer: SummaryWriter,
    train_results: dict[str, float],
    test_results: dict[str, float],
    epoch: int,
) -> None:
    """Writes matched train/test metrics to TensorBoard, grouped by metric name.

    Expects keys prefixed with "train_"/"test_" (e.g. "train_ce", "test_ce") and
    pairs them by their shared suffix, so each metric gets its own graph
    comparing train vs. test over epochs.

    Args:
      writer: SummaryWriter to log to.
      train_results: Dict of training metrics, keys prefixed with "train_".
      test_results: Dict of testing metrics, keys prefixed with "test_".
      epoch: Current epoch, used as the x-axis step.

    Raises:
      ValueError: If train_results and test_results don't cover the same
        set of metric names (after stripping the train_/test_ prefix).
    """
    train_metrics = {
        name.removeprefix("train_"): value for name, value in train_results.items()
    }
    test_metrics = {
        name.removeprefix("test_"): value for name, value in test_results.items()
    }

    if train_metrics.keys() != test_metrics.keys():
        raise ValueError(
            f"train_results and test_results must cover the same metrics. "
            f"Got {sorted(train_metrics.keys())} and {sorted(test_metrics.keys())}."
        )

    for name in train_metrics:
        write_scalar(
            writer,
            tag_scalar_dict={"train": train_metrics[name], "test": test_metrics[name]},
            main_tag=name,
            epoch=epoch,
        )


# def print_epoch_stats(epoch, train_losses, test_losses):
#     print(
#         f"Epoch: {epoch+1} | "
#         f"train_loss: {train_total:.4f} | "
#         f"train_acc: {train_ce:.4f} | "
#         f"test_loss: {train_l1:.4f} | "
#         f"test_acc: {train_l2:.4f}"
#     )

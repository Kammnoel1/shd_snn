from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from snn_shd import config
from snn_shd.utils import get_data_paths


class SHDDataset(torch.utils.data.Dataset):
    """
    SHD dataset returning dense spike tensors for snnTorch.

    Args:
      data_path: Path to the SHD .h5 file (training or test set).
      duration: Simulation duration in seconds.
      num_steps: Number of discrete time steps to bin spikes into.
      num_neurons: Number of input neurons.
    """

    def __init__(
        self, data_path: Path, duration: float, num_steps: int, num_neurons: int
    ):
        self.data_file = data_path
        self.num_neurons = num_neurons
        self.num_steps = num_steps
        self.max_time = duration

        with h5py.File(data_path, "r") as f:
            self.labels = torch.from_numpy(f["labels"][:]).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Returns a binned spike matrix and its label for a given index.

        Args:
          idx: Index of the sample to retrieve.

        Returns:
          A tuple of (spike_matrix, label), where spike_matrix has shape
          (num_steps, num_neurons) and label is a scalar tensor.
        """
        with h5py.File(self.data_file, "r") as f:
            times = f["spikes/times"][idx]
            units = f["spikes/units"][idx]

        label = self.labels[idx]

        time_bins = np.linspace(
            0, self.max_time, num=self.num_steps + 1
        )  # Create num_steps bins
        bin_indices = np.clip(
            np.digitize(times, time_bins) - 1, 0, self.num_steps - 1
        )  # np.digitize is 1-based, substract 1 to get 0-based bins
        spike_matrix = torch.zeros(
            self.num_steps, self.num_neurons, dtype=torch.float32
        )

        spike_matrix[bin_indices, units] = 1.0

        return spike_matrix, label


def create_dataloaders(
    data_path: Path,
    device: torch.device,
    duration: float = config.DURATION,
    num_steps: int = config.NUM_STEPS,
    num_neurons: int = config.IN_NEURONS,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> tuple[DataLoader, DataLoader]:
    """
    Creates training and testing DataLoaders for the SHD dataset.

    Args:
      data_path: Path to the SHD data directory.
      device: Target device; used to decide whether to pin memory.
      duration: Simulation duration in seconds.
      num_steps: Number of discrete time steps to bin spikes into.
      num_neurons: Number of input neurons.
      batch_size: Number of samples per batch in each DataLoader.
      num_workers: Number of subprocesses used for data loading.

    Returns:
      A tuple of (train_dataloader, test_dataloader).
    """
    pin_memory = device.type == "cuda"

    train_path, test_path = get_data_paths(data_path)
    train_data = SHDDataset(
        data_path=train_path,
        duration=duration,
        num_steps=num_steps,
        num_neurons=num_neurons,
    )
    test_data = SHDDataset(
        data_path=test_path,
        duration=duration,
        num_steps=num_steps,
        num_neurons=num_neurons,
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,  # don't need to shuffle test data
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataloader, test_dataloader

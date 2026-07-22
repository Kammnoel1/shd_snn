import torch
from torch.utils.data import DataLoader
import h5py
import numpy as np
from pathlib import Path


class SHDDataset(torch.utils.data.Dataset):
    """
    Needs to implemented correctly. Should take train_dir, test_dir as well as train_transform and test_transform as inputs.
    Should have .classes attribute to return the names of the classes.
    """

    # def __init__(self, data_dict, duration, n_steps, n_neurons, split=None):
    #     self.num_neurons = n_neurons
    #     self.num_steps = n_steps
    #     self.max_time = duration
    #     if split == "train":
    #         self.data_file = Path(data_dict) / "shd_train.h5"
    #     else:
    #         self.data_file = Path(data_dict) / "shd_test.h5"

    #     with h5py.File(self.data_file, "r") as f:
    #         self.labels = torch.from_numpy(f["labels"][:]).long()

    # def __len__(self):
    #     return len(self.labels)

    # def __getitem__(self, idx):
    #     with h5py.File(self.data_file, "r") as f:
    #         times = f["spikes/times"][idx]
    #         units = f["spikes/units"][idx]

    #     label = self.labels[idx]

    #     time_bins = np.linspace(
    #         0, self.max_time, num=self.num_steps + 1
    #     )  # create 2000 bins
    #     bin_indices = np.clip(
    #         np.digitize(times, time_bins) - 1, 0, self.num_steps - 1
    #     )  # np.digitize is 1-based, substract 1 to get 0-based bins
    #     spike_matrix = torch.zeros(
    #         self.num_steps, self.num_neurons, dtype=torch.float32
    #     )

    #     spike_matrix[bin_indices, units] = 1.0

    #     return spike_matrix, label
    pass


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    train_transform: None,
    test_transform: None,
    batch_size: int,
    num_workers: int,
):
    """ """
    train_data = SHDDataset(train_dir=train_dir, transform=train_transform)
    test_data = SHDDataset(test_dir=test_dir, transform=test_transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,  # don't need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names

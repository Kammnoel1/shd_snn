from pathlib import Path

import torch

from snn_shd import config


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


def get_data_paths(dir_path: Path) -> tuple[Path, Path]:
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
    if not dir_path.is_dir():
        raise NotADirectoryError(f"The path '{dir_path}' is not a directory.")
    train_path = dir_path / config.TRAIN_FILENAME
    test_path = dir_path / config.TEST_FILENAME

    if not train_path.is_file():
        raise FileNotFoundError(f"The file '{train_path}' does not exist.")
    if not test_path.is_file():
        raise FileNotFoundError(f"The file '{test_path}' does not exist.")
    return train_path, test_path

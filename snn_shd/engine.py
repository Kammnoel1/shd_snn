from collections.abc import Callable

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

from snn_shd import config
from snn_shd.utils import MetricTracker, write_results


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    optimizer: Optimizer,
    device: torch.device,
):
    """ """
    model.train()

    tracker = MetricTracker()

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        mem2_trace, spk1_trace = model(X)

        ce_loss, l1_loss, l2_loss = loss_fn(mem2_trace, spk1_trace, labels=y)

        total_loss = ce_loss + l1_loss + l2_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        tracker.update(
            batch_size=X.shape[0],
            train_total=total_loss.item(),
            train_ce=ce_loss.item(),
            train_l1=l1_loss.item(),
            train_l2=l2_loss.item(),
        )

    return tracker.get_running_avg()


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    device: torch.device,
):
    """"""
    model.eval()
    tracker = MetricTracker()

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            mem2_trace, spk1_trace = model(X)
            ce_loss, l1_loss, l2_loss = loss_fn(mem2_trace, spk1_trace, labels=y)
            total_loss = ce_loss + l1_loss + l2_loss
            tracker.update(
                batch_size=X.shape[0],
                test_total=total_loss.item(),
                test_ce=ce_loss.item(),
                test_l1=l1_loss.item(),
                test_l2=l2_loss.item(),
            )

    return tracker.get_running_avg()


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    loss_fn: Callable,
    optimizer: Optimizer,
    device: torch.device,
    writer: SummaryWriter | None = None,
    epochs: int = config.NUM_EPOCHS,
) -> dict[str, list]:
    """
    Trains and evaluates a model for a number of epochs, logging metrics.

    Args:
      model: The PyTorch model to train and evaluate.
      train_dataloader: DataLoader for the training set.
      test_dataloader: DataLoader for the test set.
      loss_fn: Callable returning (ce_loss, l1_loss, l2_loss).
      optimizer: Optimizer used to update model parameters.
      epochs: Number of epochs to train for.
      device: Target device to compute on.
      writer: SummaryWriter for TensorBoard logging, or None to disable logging.

    Returns:
      A dict mapping each metric name (e.g. "train_ce", "test_l1") to a
      list of its value at each epoch.
    """
    model = model.to(device)
    results: dict[str, list] = {}

    for epoch in tqdm(range(epochs)):
        train_results = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_results = evaluate(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        for name, value in {**train_results, **test_results}.items():
            results.setdefault(name, []).append(value)

        if writer is not None:
            write_results(writer, train_results, test_results, epoch)

    if writer is not None:
        writer.close()

    return results

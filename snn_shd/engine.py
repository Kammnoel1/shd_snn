import sys
import time
from collections.abc import Callable

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

from snn_shd import config
from snn_shd.utils import MetricTracker, print_epoch_stats, save_model, write_results


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

        class_preds = mem2_trace.max(dim=0).values.argmax(dim=1)
        train_acc = (class_preds == y).float().mean().item()

        tracker.update(
            batch_size=X.shape[0],
            train_total=total_loss.item(),
            train_ce=ce_loss.item(),
            train_l1=l1_loss.item(),
            train_l2=l2_loss.item(),
            train_acc=train_acc,
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

            class_preds = mem2_trace.max(dim=0).values.argmax(dim=1)
            test_acc = (class_preds == y).float().mean().item()

            tracker.update(
                batch_size=X.shape[0],
                test_total=total_loss.item(),
                test_ce=ce_loss.item(),
                test_l1=l1_loss.item(),
                test_l2=l2_loss.item(),
                test_acc=test_acc,
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
    checkpoint_dir: str | None = None,
    model_name: str = config.MODEL_NAME,
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
      checkpoint_dir: Directory to save a checkpoint after each epoch, or None to disable per-epoch checkpointing.
      model_name: Filename (ending in .pt) for the saved checkpoint.

    Returns:
      A dict mapping each metric name (e.g. "train_ce", "test_l1") to a
      list of its value at each epoch.
    """
    model = model.to(device)
    results: dict[str, list] = {}

    for epoch in tqdm(range(epochs), disable=not sys.stdout.isatty()):
        epoch_start = time.perf_counter()
        train_results = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        epoch_time = time.perf_counter() - epoch_start
        test_results = evaluate(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        for name, value in {**train_results, **test_results}.items():
            results.setdefault(name, []).append(value)
        results.setdefault("epoch_time_sec", []).append(epoch_time)

        print_epoch_stats(
            epoch=epoch,
            train_results=train_results,
            test_results=test_results,
            epoch_time=epoch_time,
        )

        if writer is not None:
            write_results(writer, train_results, test_results, epoch)
            writer.add_scalar("epoch_time", epoch_time, epoch)

        if checkpoint_dir is not None:
            save_model(
                model=model,
                target_dir=checkpoint_dir,
                model_name=f"{model_name}_epoch_{epoch+1:03d}.pt",
            )

    if writer is not None:
        writer.close()

    return results

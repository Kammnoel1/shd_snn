import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from snn_shd.losses import max_over_time_loss, regularization_loss


def train_one_epoch(model, optimizer, train_loader, device):
    """
    Needs to be implemented. Cf.: https://github.com/pytorch/vision/blob/main/references/classification/train.py.
    """
    # model = model.to(device)

    #     epoch_start = time.perf_counter()
    #     model.train()
    #     epoch_ce, epoch_l1, epoch_l2 = 0.0, 0.0, 0.0
    #     n_samples = 0

    #     for x, y in train_loader:
    #         x, y = x.to(device), y.to(device)

    #         mem2_trace, spk1_trace = model(x)

    #         L_ce = max_over_time_loss(mem2_trace, y)
    #         L1, L2 = regularization_loss(spk1_trace)
    #         loss = L_ce + L1 + L2

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         epoch_ce += L_ce.item() * x.shape[0]
    #         epoch_l1 += L1.item() * x.shape[0]
    #         epoch_l2 += L2.item() * x.shape[0]
    #         n_samples += x.shape[0]
    #     epoch_time = time.perf_counter() - epoch_start
    #     print(
    #         f"Epoch {epoch+1:3d}/{N_EPOCHS} | "
    #         f"CE: {epoch_ce/n_samples:.4f} | "
    #         f"L1: {epoch_l1/n_samples:.4f} | "
    #         f"L2: {epoch_l2/n_samples:.4f} | "
    #         f"Time: {epoch_time:.2f}"
    #     )


def evaluate():
    pass


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
) -> Dict[str, List]:
    """
    Needs to be implemented. Add tensorboard SummaryWriter() for experiment tracking.
    """


#   results = {"train_loss": [],
#       "train_acc": [],
#       "test_loss": [],
#       "test_acc": []
#   }

#   # Loop through training and testing steps for a number of epochs
#   for epoch in tqdm(range(epochs)):
#       train_loss, train_acc = train_one_epoch(model=model,
#                                           dataloader=train_dataloader,
#                                           loss_fn=loss_fn,
#                                           optimizer=optimizer,
#                                           device=device)
#       test_loss, test_acc = test_step(model=model,
#           dataloader=test_dataloader,
#           loss_fn=loss_fn,
#           device=device)

#       # Print out what's happening
#       print(
#           f"Epoch: {epoch+1} | "
#           f"train_loss: {train_loss:.4f} | "
#           f"train_acc: {train_acc:.4f} | "
#           f"test_loss: {test_loss:.4f} | "
#           f"test_acc: {test_acc:.4f}"
#       )

#       # Update results dictionary
#       results["train_loss"].append(train_loss)
#       results["train_acc"].append(train_acc)
#       results["test_loss"].append(test_loss)
#       results["test_acc"].append(test_acc)

#   # Return the filled results at the end of the epochs
#   return results

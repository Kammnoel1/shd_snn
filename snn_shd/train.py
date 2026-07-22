from pathlib import Path
import torch
import data_setup, engine, feedforward_simple, utils
from snn_shd import config

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = config.DATA_DICT / "train"
test_dir = config.DATA_DICT / "test"


train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=None,
    batch_size=config.BATCH_SIZE,
)


model = feedforward_simple.FeedforwardSNN()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=config.LEARNING_RATE)

engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=config.NUM_EPOCHS,
    device=device,
)

utils.save_model(
    model=model,
    target_dir="models",
    model_name="05_going_modular_script_mode_tinyvgg_model.pth",
)

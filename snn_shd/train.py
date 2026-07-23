import torch

from snn_shd import config, data_setup, engine, feedforward_simple, losses, utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader, test_dataloader = data_setup.create_dataloaders(
    device=device,
)


model = feedforward_simple.FeedforwardSNN()

loss_fn = losses.combined_loss
optimizer = torch.optim.Adamax(params=model.parameters(), lr=config.LEARNING_RATE)
writer = utils.create_writer(experiment_name="test_runs", model_name=config.MODEL_NAME)
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
    writer=writer,
)

utils.save_model(
    model=model,
    target_dir="models",
    model_name=config.MODEL_NAME,
)

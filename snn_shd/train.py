import torch

from snn_shd import config, data_setup, engine, feedforward_simple, utils

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    data_path=config.DATA_DIR,
    device=device,
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
    model_name=config.MODEL_NAME,
)

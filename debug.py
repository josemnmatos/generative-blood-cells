# %%
from torch.nn import functional as F
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from custom_models.cvae import CVAE, VAELoss
from custom_models.vae import VariationalAutoencoder
from generative_pipeline import CustomGenerativePipeline
from utils import load_medMNIST_data
import os
import sys


# %%

# %%
dataloader, dataset = load_medMNIST_data()

# %%
# get image size from the dataset
channels, dim_1, dim_2 = dataloader.dataset[0][0].shape

print(f"Channels: {channels}")
print(f"Dim 1: {dim_1}")
print(f"Dim 2: {dim_2}")

# %%
# ********** Model * **********
model = CVAE(
    input_channels=channels,
    latent_dim=32,
    output_channels=channels,
)


# ********** Training parameters **********
loss_fn = VAELoss(kld_weight=0.5)
optimizer_class = torch.optim.Adam
lr = 1e-4
n_epochs = 10

model

# %%
pipeline = CustomGenerativePipeline(
    model=model,
    criterion=loss_fn,
    optimizer_class=optimizer_class,
    optimizer_params={"lr": lr},
    train_dataloader=dataloader,
    train_dataset=dataset,
    n_epochs=n_epochs,
)

pipeline.execute()
pipeline.visualize(30)
pipeline.evaluate()

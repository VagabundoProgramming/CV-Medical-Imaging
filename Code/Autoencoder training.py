import numpy as np
import pandas as pd
import cv2
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from Autoencoder import *

#If this cell fails you need to change the runtime of your colab notebook to GPU
# Go to Runtime -> Change Runtime Type and select GPU
assert torch.cuda.is_available(), "GPU is not enabled"

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images_path = get_img_paths("Cropped", "PatientDiagnosis.csv")

train_loader = DataLoader(Cropped_Patches(images_paths, data_type="train"), batch_size=64, shuffle=True)
test_loader = DataLoader(Cropped_Patches(images_paths, data_type="test"), batch_size=64, shuffle=True)

model = AE(bottleneck=100).to(device)

# create an optimizer object
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()

epochs = 100
losses = {"train":[], "test":[]}
for epoch in range(epochs):
    losses["train"].append(train(model, train_loader, optimizer, criterion, epoch))
    losses["test"].append(test(model, test_loader, criterion, epoch))
    # Every epoch it creates a plot of the loss in the train set and validation set, this plot is saved
    plt.plot(losses["train"], label="training loss")
    plt.plot(losses["test"], label="validation loss")
    plt.savefig("figures/nothing_loss"+str(epoch)+".png")
    plt.pause(0.0001)
    plt.clf()

print(losses)
print("DONE!")

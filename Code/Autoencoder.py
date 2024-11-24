import numpy as np
import pandas as pd
import cv2
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

class AE(nn.Module):
    def __init__(self, bottleneck=100):
        super(AE, self).__init__()
        # Encoder section
        self.Conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.Pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.relu1 = nn.ReLU()
        self.Conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.Pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.relu2 = nn.ReLU()
        self.Conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.Pool3 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.relu3 = nn.ReLU()
        self.Conv4 = nn.Conv2d(32, 64, 3)
        self.Pool4 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.relu4 = nn.ReLU()
        self.Linear = nn.Linear(14400, bottleneck)
        self.relu5 = nn.ReLU()
        # Decoder section
        self.revLinear = nn.Linear(bottleneck, 14400)
        self.relu6 = nn.ReLU()
        self.UnPool1 = nn.MaxUnpool2d(2, stride=2)
        self.revConv1 = nn.ConvTranspose2d(64, 32, 3)
        self.relu7 = nn.ReLU()
        self.UnPool2 = nn.MaxUnpool2d(2, stride=2)
        self.revConv2 = nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.relu8 = nn.ReLU()
        self.UnPool3 = nn.MaxUnpool2d(2, stride=2)
        self.revConv3 = nn.ConvTranspose2d(16, 8, 3, padding=1)
        self.relu9 = nn.ReLU()
        self.UnPool4 = nn.MaxUnpool2d(2, stride=2)
        self.revConv4 = nn.ConvTranspose2d(8, 3, 3, padding=1)
        self.tanh = nn.Tanh()


    def forward(self, x):
        batch_size = x.shape[0]
        #Encoder - Conv
        x = self.Conv1(x)
        x, indices1 = self.Pool1(x)
        x = self.relu1(x)
        x = self.Conv2(x)
        x, indices2 = self.Pool2(x)
        x = self.relu2(x)
        x = self.Conv3(x)
        x, indices3 = self.Pool3(x)
        x = self.relu3(x)
        x = self.Conv4(x)
        x, indices4 = self.Pool4(x)
        x = self.relu4(x)
        #Encoder - Linear
        x = torch.reshape(x, (batch_size, 14400))
        x = self.Linear(x)
        x = self.relu5(x)
        #Decoder - Linear
        x = self.revLinear(x)
        x = self.relu6(x)
        x = torch.reshape(x, (batch_size, 64, 15, 15))
        #Decoder - Conv
        x = self.UnPool1(x, indices4)
        x = self.revConv1(x)
        x = self.relu7(x)
        x = self.UnPool2(x, indices3)
        x = self.revConv2(x)
        x = self.relu8(x)
        x = self.UnPool3(x, indices2)
        x = self.revConv3(x)
        x = self.relu9(x)
        x = self.UnPool4(x, indices1)
        x = self.revConv4(x)
        x = self.tanh(x)
        return x

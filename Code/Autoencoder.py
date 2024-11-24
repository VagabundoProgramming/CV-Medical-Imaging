import numpy as np
import pandas as pd
import cv2
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

def get_img_paths(images_folder, PatientDiagnosis):
    images_paths = []
    patients = np.array(os.listdir(images_folder))
    df = pd.read_csv(PatientDiagnosis)
    df = df[df["DENSITAT"] == "NEGATIVA"]
    pos_folders = np.concatenate((df["CODI"].to_numpy()+"_0", df["CODI"].to_numpy()+"_1"))
    folders = np.intersect1d(patients, pos_folders)
    for fold in folders:
        for name in os.listdir(images_folder + "/" + fold):
            if name[-3:] == "png":
                images_paths.append(images_folder + "/" + fold + "/" + name)
    return images_paths

class Cropped_Patches(Dataset):
    def __init__(self, images_path, data_type="train"):
        if data_type == "train":
            self.img_paths = images_path[:10000]
        elif data_type == "test":
            self.img_paths = images_path[-1000:]
        else:
            raise TypeError

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, key):
        name = self.img_paths[key]
        img = cv2.imread(name)
        if img.shape != (256, 256, 3):
            img = cv2.resize(img, (256,256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 1, 0))
        img = torch.from_numpy(img).to(torch.float)
        return img / 255

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

def save_process(img, name):
    img = torch.transpose(img, 1, 2)
    save_image(img, "figures/"+name)

def train(model, loader, optimizer, criterion, epoch):
    loss = 0
    model.train()

    for batch_features in loader:
        # load it to the active device
        batch_features = batch_features.to(device)
        
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        
        # compute reconstructions
        outputs = model(batch_features)
        
        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)
        
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(loader)
    print("epoch : {}/{}, Train loss = {:.6f}".format(epoch + 1, epochs, loss))
    if epoch % 1 == 0:
        save_process(torch.cat(
            (make_grid(batch_features.detach().cpu().view(-1, 3, 256, 256).transpose(2, 3), nrow=2, normalize = True),
            make_grid(outputs.detach().cpu().view(-1, 3, 256, 256).transpose(2, 3), nrow=2, normalize = True)),
            2), "new10_train"+str(epoch)+".png")
    return loss

def test(model, loader, criterion, epoch):
    loss = 0
    model.eval()
    
    for batch_features in loader:
        batch_features = batch_features.to(device)

        with torch.no_grad():
            outputs = model(batch_features)
        
        # compute training reconstruction loss
        test_loss = criterion(outputs, batch_features)
 
        # add the mini-batch training loss to epoch loss
        loss += test_loss.item()
    
    # compute the epoch test loss
    loss = loss / len(loader)
    
    # display the epoch training loss
    print("epoch : {}/{}, Test loss = {:.6f}".format(epoch + 1, epochs, loss))
    if epoch % 1 == 0:
        save_process(torch.cat(
            (make_grid(batch_features.detach().cpu().view(-1, 3, 256, 256).transpose(2, 3), nrow=2, normalize = True),
            make_grid(outputs.detach().cpu().view(-1, 3, 256, 256).transpose(2, 3), nrow=2, normalize = True)),
            2), "new10_test"+str(epoch)+".png")
        torch.save(model.state_dict(), "SavedModels/new10_model_epoch"+str(epoch)+".pt")
    return loss

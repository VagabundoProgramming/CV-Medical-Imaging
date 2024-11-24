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

class Cropped_Patches(Dataset):
    def __init__(self, images_folder, CoordPatches):
        df = pd.read_excel(CoordPatches)
        #df = df[df["Presence"] == -1]
        df = df.to_numpy()
        self.img_paths = ["-"] * len(df)
        self.presence = [0] * len(df)
        for i in range(len(df)):
            name = images_folder + "/" + df[i,0] + "_" + str(df[i,1]) + "/" + "0"*(5-len(str(df[i,2]))) + str(df[i,2]) + ".png"
            self.img_paths[i] = name
            self.presence[i] = df[i,-1]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, key):
        img = cv2.imread(self.img_paths[key])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).to(torch.float), self.presence[key]

CP = Cropped_Patches("Annotated", "HP_WSI-CoordAnnotatedPatches.xlsx")

model = AE(bottleneck=100).to(device)
model.load_state_dict(torch.load("SavedModels/model_100_epoch52.pt", weights_only=True))
#Since we trained different models we saved each model with its bottleneck on its name
#Therefore in this case the _100_ refers to a bottleneck of 100 neuron
#Remind to change its bottleneck to the one referred on its name
#Best model 52

criterion = nn.MSELoss()

res = []
for i, (x, l) in enumerate(CP):
    x1 = x / 255
    x = x1.unsqueeze(0)
    x = np.transpose(x, (0, 3, 2, 1))
    x = x.to(device)
    with torch.no_grad():
       x2 = (np.transpose(model(x).to("cpu").squeeze(), (2, 1, 0)))
    er = criterion(x1, x2)
    res.append([er, l])
res = np.array(res)

with_P = res[res[:,1] == 1][:,0]
without_P = res[res[:,1] == -1][:,0]
unknown_P = res[res[:,1] == 0][:,0]

dist = [with_P, without_P, unknown_P]

labels = ["with_P", "without_P", "unknown_P"]

fig, ax = plt.subplots()
ax.set_ylabel('Density Pylori')
 
bplot = ax.boxplot(dist, patch_artist=True,)
plt.xticks([1, 2, 3], labels)
plt.show()

best_acc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
best_F1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
for threshold in np.linspace(with_P.mean(), without_P.mean(), 100):
    P = res[res[:,0] > threshold]
    N = res[res[:,0] <= threshold]
    TP = len(P[P[:,1] == 1])
    FN = len(N[N[:,1] == 1])
    FP = len(P[P[:,1] == -1])
    TN = len(N[N[:,1] == -1])
    acc = (TP+TN)/(TP+FN+FP+TN) #Accuracy
    if TP+FP != 0: pre = TP/(TP+FP) #Precision
    else: pre = 0
    if TP+FN != 0: rec = TP/(TP+FN) #Recall - Sensitivity
    else: rec = 0
    if TN+FP != 0: spe = TN/(TN+FP) #Specificity
    else: spe = 0
    if pre+rec != 0: F1 = (2*pre*rec)/(pre+rec) #F1 Score
    else: F1 = 0
    if acc > best_acc[1]:
        best_acc = [threshold, acc, pre, rec, spe, F1]
    if F1 > best_F1[5]:
        best_F1 = [threshold, acc, pre, rec, spe, F1]
    #print("Accuracy:", "%.3f"%acc, "\tPrecision:", "%.3f"%pre, "\tRecall:", "%.3f"%rec, "\tSpecificity:", "%.3f"%spe, "\tF1 Score:", "%.3f"%F1)
#If we take as everyting without Pypol then our accuracy grew up to 0.865
print("\nBest Accuracy Threshold:", best_acc[0], "\n\tAccuracy:", "%.3f"%best_acc[1], "\tPrecision:", "%.3f"%best_acc[2], "\tRecall:", "%.3f"%best_acc[3], "\tSpecificity:", "%.3f"%best_acc[4], "\tF1 Score:", "%.3f"%best_acc[5])
print("Best F1 Score Threshold:", best_F1[0], "\n\tAccuracy:", "%.3f"%best_F1[1], "\tPrecision:", "%.3f"%best_F1[2], "\tRecall:", "%.3f"%best_F1[3], "\tSpecificity:", "%.3f"%best_F1[4], "\tF1 Score:", "%.3f"%best_F1[5])

PatientDiagnosis = "PatientDiagnosis.csv"
images_folder = "HoldOut"

df = pd.read_csv(PatientDiagnosis)
df = df.to_numpy()
patients = []
for i, (fold, diagnosis) in enumerate(df):
    print(str(i)+"/"+str(df.shape[0]))
    patches_er = []
    for added in ["_0", "_1"]:
        if not os.path.exists(images_folder+"/"+fold+added):
            continue
        for name in os.listdir(images_folder+"/"+fold+added):
            if name[-3:] == ".db":
                continue
            img = cv2.imread(images_folder+"/"+fold+added+"/"+name)
            if img.shape != (256, 256, 3):
                img = cv2.resize(img, (256, 256))
            img = np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255, (2, 1, 0))
            img = torch.from_numpy(img).to(device, dtype=torch.float).unsqueeze(0)
            er = criterion(model(img), x).to("cpu").detach().item()
            patches_er.append(er)
    if patches_er != []:
        patients.append(np.array(patches_er).mean())
    else:
        patients.append("-")
patients = np.array(patients)
df = df[patients != "-"]
patients = patients[patients != "-"]
patients = patients.astype("float64")

best_acc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
best_F1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for threshold in np.linspace(patients.min(), patients.max(), 1000):
    P = df[patients > threshold]
    N = df[patients <= threshold]
    TP = len(P[P[:,1] != "NEGATIVA"])
    FN = len(N[N[:,1] != "NEGATIVA"])
    FP = len(P[P[:,1] == "NEGATIVA"])
    TN = len(N[N[:,1] == "NEGATIVA"])
    if TP+FP != 0: acc = (TP+TN)/(TP+FN+FP+TN) #Accuracy
    else: acc = 0
    if TP+FP != 0: pre = TP/(TP+FP) #Precision
    else: pre = 0
    if TP+FN != 0: rec = TP/(TP+FN) #Recall - Sensitivity
    else: rec = 0
    if TN+FP != 0: spe = TN/(TN+FP) #Specificity
    else: spe = 0
    if pre+rec != 0: F1 = (2*pre*rec)/(pre+rec) #F1 Score
    else: F1 = 0
    if acc > best_acc[1]:
        best_acc = [threshold, acc, pre, rec, spe, F1]
    if F1 > best_F1[5]:
        best_F1 = [threshold, acc, pre, rec, spe, F1]
    #print("Accuracy:", "%.3f"%acc, "\tPrecision:", "%.3f"%pre, "\tRecall:", "%.3f"%rec, "\tSpecificity:", "%.3f"%spe, "\tF1 Score:", "%.3f"%F1)
print("\nBest Accuracy Threshold:", best_acc[0], "\n\tAccuracy:", "%.3f"%best_acc[1], "\tPrecision:", "%.3f"%best_acc[2], "\tRecall:", "%.3f"%best_acc[3], "\tSpecificity:", "%.3f"%best_acc[4], "\tF1 Score:", "%.3f"%best_acc[5])
print("Best F1 Score Threshold:", best_F1[0], "\n\tAccuracy:", "%.3f"%best_F1[1], "\tPrecision:", "%.3f"%best_F1[2], "\tRecall:", "%.3f"%best_F1[3], "\tSpecificity:", "%.3f"%best_F1[4], "\tF1 Score:", "%.3f"%best_F1[5])

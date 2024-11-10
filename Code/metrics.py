# This file defines a bunch of functions to evaluate our models

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from load_data import data_splitter, Cropped_Patches


# Creates a graph displaying loss and accuracy of the model
def display_loss_acc(loss_l, acc_l, model_name):
    y_loss = loss_l
    x_loss = [x for x in range (0, len(loss_l), 1)]

    y_acc = acc_l
    x_acc = x_loss

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(x_loss, y_loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  

    color = 'tab:red'
    ax2.set_ylabel('Accuracy', color=color)  
    ax2.plot(x_acc, y_acc, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    fig.savefig(model_name+"_Loss_Acc.png")
    plt.show()

# Creates a confusion matrix of the model
def display_conf_m(dataloader, model, model_name):
    model.eval()
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            #X, y = X.to(device), y.to(device)
            pred = model(X)

            for p, sub_y in zip(pred, y):
                
                if p[0] < p[1]:   # (0,1) == infected
                    sub_p = True
                else:
                    sub_p = False
                
                if sub_y[0] < sub_y[1]:   # (0,1) == infected
                    sub_y = True
                else:
                    sub_y = False
                
                # Update values
                if sub_p == sub_y == True:
                    TP += 1
                elif sub_p == sub_y == False:
                    TN += 1
                elif sub_p != sub_y and sub_p == True:
                    FP += 1
                else:
                    FN += 1
    full_conf = np.array([[TP, FP], [FN, TN]])

    labels = ["Infected", "Healthy"]

    fig, ax = plt.subplots(ncols = 1, figsize = (12, 12))

    
    disp = ConfusionMatrixDisplay(full_conf, display_labels = labels)
        
    disp.plot(ax=ax)
    disp.ax_.set_title("Confusion Matrix of the Model", fontdict = {"fontsize" : 25}, pad = 1)

    fig.savefig(model_name+"_Confusion_Matrix.png")
    plt.show()

# Returns the accuracy on different partititons of data and the average accuracy
def average_accuracy(model, loss_fn, test, all_images, splits, model_name):
    acc_l = []

    for x in range(0, splits, 1):
        # Define test_dataset
        train_set, test_set = data_splitter(all_images, 5, x)
        del train_set
        test_dataset  = Cropped_Patches(test_set)
        test_loader  = DataLoader(test_dataset, batch_size = 20, shuffle = False)
        
        # Caclulate Accuracy
        loss, acc = test(test_loader, model, loss_fn)
        del loss
        acc_l.append(acc)
    
    avg_acc = sum(acc_l) / len(acc_l)
    std_acc = np.std(acc_l)
    return acc_l, avg_acc, std_acc

    print(f"The model {model_name}")
    print(f"With the accuracities:\n {acc_l}\n")
    print(f"Has an average accuracy of: {avg_acc}")
    print(f"And a standard deviation of: {std_acc}")



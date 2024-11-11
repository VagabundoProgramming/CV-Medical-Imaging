# This file contains the A OJO model
# It defines everything necessary to create the datasets, dataloader, 
# model, train, test and evaluation.

# This model does not require training

import matplotlib.pyplot as plt
import numpy as np
from load_data import *
from metrics import *

### Settings
# If false skips the calculation of optimal threshold and related functions
prove_optimal = False

### Load data and clean it up ###
data_path = "Data\\"
annotations, patch_data = open_data(data_path)
ids, ids_mod = get_ids(image_path="Data\\Annotated") 

# Clean Data
cleaned_patches = clean_patches(patch_data, ids_mod)

# Load Images
all_images = load_images(cleaned_patches) 

# Split Data
train_set, test_set = data_splitter(all_images, 5, 0)

# Define Datasets and Dataloaders
dataset = Cropped_Patches(all_images)

### Model 
def model(img, threshold = 25):
    img = img.transpose(0,1)
    temp = (img[0][0][0], img[0][1][0], img[0][2][0])

    for x in range(1, len(img[0]), 1):
        for y in range (0, len(img[0][0]), 1):
            temp2 = (float(img[0][x][y]), float(img[1][x][y]), float(img[2][x][y]))
            if y != 0 and threshold < (temp[0]-temp2[0])+(temp[1]-temp2[1])+(temp[2]-temp2[2]):
                return torch.tensor([0,1])
            temp = temp2
    return torch.tensor([1,0])

### Test
def test (dataset, model, threshold = 210):
    size = len(dataset)
    num_batches = len(dataset)

    test_loss, correct = 0,0
    
    for X, y in dataset:
        pred = model(X, threshold) # The optimal threshold is 210, as reasoned below
        if pred[0] == y[0] and pred[1] == y[1]:
            correct += 1

    test_loss /= num_batches
    correct /=size
    return correct, threshold
    

acc, threshold = test(dataset, model)
print(f"\"A ojo\" model: \n Accuracy: \t{(100*acc):>0.1f}% \t Threshold: {threshold}\n")

### Roc Curve

# Test for optimal threshold
def ROC_curve(dataset, model, start, end, step):
    # Distance function
    def distance2D(pointA, pointB):
        return((pointA[0]-pointB[0])**2 + (pointA[1]-pointB[1])**2)**0.5

    ROC_data=[]
    for threshold in range(start, end, step):
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        e = 0.0001

        for X, y in dataset:
            pred = model(X, threshold = threshold) #58 current optimal
            # Assuming Positive means infected
            if y[0] == 0 and y[1] == 1: # Pred is positive
                if pred[0] == 0 and pred[1] == 1:
                    TP += 1
                if pred[0] == 1 and pred[1] == 0:
                    FN += 1
            if y[0] == 1 and y[1] == 0: # Pred is negative
                if pred[0] == 0 and pred[1] == 1:
                    FP += 1
                if pred[0] == 1 and pred[1] == 0:
                    TN += 1                         
        
        TPR = TP/(TP+FN+e)
        FPR = FP/(TN+FP+e)
        ROC_data.append({"threshold" : threshold, "TPR": TPR, "FPR": FPR, "dist" : distance2D((0,1), (FPR, TPR))})

    return ROC_data

# Reorders roc data for display
def order_roc_data(roc_data):
    def sort_f(dictionary):
        return(dictionary[1])
    
    l = [[dictionary["threshold"], dictionary["dist"], dictionary["TPR"], dictionary["FPR"]] for dictionary in roc_data]
    
    l.sort(key = sort_f)
    return l

# These start, end and step values re cherrypicked
# If tested with step 1, one should arrive at 210 as the optimal anyway
if prove_optimal:
    roc_d = ROC_curve(dataset, model, 0, 481, 60)
    a = order_roc_data(roc_d)
    best = a[0]

    # Display optimal threshold
    print(f"ROC DATA RESULTS \n\t Best threshold {best[0]}")
    print(f"\t\t with TP rate of {best[2]*100:>0.5f} %")
    print(f"\t\t with FP rate of {best[3]*100:>0.5f} %")

# Display ROC curve
def order_roc_data_for_plot(roc_data):
    def sort_f(dictionary):
        return(dictionary[1])
    
    l = [[ dictionary["TPR"], dictionary["FPR"]] for dictionary in roc_data]
    
    l.sort(key = sort_f, reverse = False)
    return l

# Cannot prove without making all the calculations
if prove_optimal:
    l2 = order_roc_data_for_plot(roc_d)
    y = [dictionary[0] for dictionary in l2]
    x = [dictionary[1] for dictionary in l2]

    j = np.arange(0, 1.1, 0.2)

    plt.plot(x, y) 
    plt.plot(j, j, 'r--')
    plt.show()

### Confusion Matrix
display_conf_m_a_ojo(dataset, model, 210,"A_OJO")

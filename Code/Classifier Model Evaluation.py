# This file contains everything to test the accuarcy of the classifier model

import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt # Probablemente use eventualemnte
from load_data import *
from metrics import *

### Get data --------------------------------------###
# Get Patient diagnosis.csv
annotations = pd.read_csv("Data\PatientDiagnosis.csv")

# Load data
data_path = "Data\\"
annotations, _ = open_data(data_path)
ids, ids_mod = get_ids(image_path="Data\\HoldOut") 

# Clean Data
def clean_csv(annotations):
    annotations_cleaned = annotations.copy()

    for x in range (annotations.shape[0] - 1, -1, -1):
        if annotations_cleaned["CODI"][x] not in ids_mod:
            annotations_cleaned = annotations_cleaned.drop(x)
    return annotations_cleaned

annotations_cleaned = clean_csv(annotations)

# Simplify the dataset
def simplyfy_dataset(annotations_cleaned):
    #translation_dict= {"NEGATIVA" : -1, "BAIXA" : 0, "ALTA" : 1}
    translation_dict = {"NEGATIVA" : 0, "BAIXA" : 0, "ALTA" : 1}
    temp_holder = []

    for row in annotations_cleaned.iterrows():
        row_data = []

        row_data.append(row[1][0]) # Append Patient code
        row_data.append(translation_dict[row[1][1]]) # Append Diagnosis
        temp_holder.append(row_data)


    annotations_mod = pd.DataFrame(temp_holder)
    annotations_mod = annotations_mod.rename({0: "Patient", 1: "Diagnosis"}, axis = 1)
    return annotations_mod

annotations_mod = simplyfy_dataset(annotations_cleaned)

# Load images
def load_patient_img(ids, id, image_path = "Data\Annotated\\"):
    patient_imgs = []
    
    if id not in ids:
        return -1


    for file_name in os.listdir(image_path + id):
        if file_name[-3:] != ".db":
            patient_imgs.append(torch.transpose(torch.from_numpy(cv2.imread(image_path + id +"\\\\" + file_name)).to(torch.float), 0, 2))
            if patient_imgs[-1].shape != (3, 256, 256):
                del patient_imgs[-1]


    patient_imgs = torch.stack(patient_imgs)
    return patient_imgs

patient_imgs = load_patient_img(ids, ids[0],  image_path = "Data\HoldOut\\") 

### Define an iterator --------------------------------------###
# This will allow us to iterate over the data, but patient by patient
# instead of by individual patches
def get_patient_data(ids, annotations_mod, image_path = "Data\HoldOut\\"):
    max_len = annotations_mod.shape[0]
    n = 1

    for index, row in annotations_mod.iterrows():
        diagnosis = row["Diagnosis"]
        id = row["Patient"]
        
        # Check if they are either _0 or _1
        if os.path.isdir(image_path+id+"_0"):
            id = id+"_0"
        elif os.path.isdir(image_path+id+"_1"):
            id = id+"_1"

        # Get images
        patient_imgs = load_patient_img(ids, id, image_path)
        if n < max_len:
            yield patient_imgs, diagnosis
            n+=1
        else:
            return


### Define and initialize the model
# The same as in the Classifier mode
class M_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_epic = nn.Sequential(
            nn.Conv2d(3, 6, 5, 2, 2),
            nn.MaxPool2d(2,2),
            nn.ReLU(),

            nn.Conv2d(6, 12, 3, 2, 1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),

            nn.Conv2d(12, 6, 5, 2, 2),
            nn.MaxPool2d(2,2),
            nn.ReLU(),

            nn.Conv2d(6, 1, 3, 1, 1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(16,2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers_epic(x)
        return x
    
def start_model(model_name = None):
    if not model_name:
        model = M_model()#.to(device)
    else:
        model = torch.load(f"Models/{model_name}")#.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    return model, loss_fn, optimizer

# Load the already trained model
model_name = "classifier_model_parameters_70.pt"
model, loss_fn, optimizer = start_model(model_name = model_name)

### Metric functions --------------------------------------###

# Test acc allows us to run the model throught the patients and get
# The confusion matrix values on its predictions give a threshold
def test_acc(model,ids, annotations_mod, threshold = 0.5):
    # Create a data iterator
    patient_data_iterator = get_patient_data(ids=ids, annotations_mod=annotations_mod)

    TP, FP, FN, TN = 0, 0, 0, 0

    for X, y in patient_data_iterator:
        pred_ratios = [0,0]# healthy, infected

        # Make prediction
        predictions = model(X)

        # Calculate ratio
        for pred in predictions:
            if pred[0] < pred[1]: # if predicted as infected
                pred_ratios[1] += 1 # add to infected
            else: 
                pred_ratios[0] += 1 # add to healthy
        ratio = pred_ratios[1] / sum(pred_ratios) # Infected ratio

        # Classify result
        if ratio > threshold:
            # Prediction is "infected"
            if y == 1: # y is infected
                TP += 1 
            else: # y is healthy
                FP += 1 
        else:
            # Prediction is "healthy"
            if y == 1: # y is infected
                FN += 1 
            else: # y is healthy
                TN += 1 

    return TP, FP, FN, TN
     
### Roc Curve

# Given the data it calculates several options for thresholds, 
# returning their TPR, FPR and distance to optimality
def ROC_curve(model, ids, annotations_mod):
    def distance2D(pointA, pointB):
        return((pointA[0]-pointB[0])**2 + (pointA[1]-pointB[1])**2)**0.5

    ROC_data={} 
    for threshold in range(10, 100, 20):
        threshold /= 100
        print(f"Calculating for threshold {threshold} value")

        TP, FP, FN, TN = test_acc(model, ids, annotations_mod, threshold = threshold)

        TPR = TP/(TP+FN+0.0001)
        FPR = FP/(TN+FP+0.0001)
        ROC_data[threshold] = {"threshold" : threshold, "TPR": TPR, "FPR": FPR, "dist" : distance2D((0,1), (FPR, TPR))}
    return ROC_data

roc_d = ROC_curve(model, ids, annotations_mod)

# Reorders the data from ROC curve by the 
def order_roc_data(roc_data):
    def sort_f(list_obj):
        return(list_obj[1])

    ordered_data = []
    for key, roc_sub_dict in roc_data.items():
        l = [roc_sub_dict["threshold"], roc_sub_dict["dist"], roc_sub_dict["TPR"], roc_sub_dict["FPR"]]
        ordered_data.append(l)
    ordered_data.sort(key = sort_f)
    return ordered_data

ordered_ROC_data = order_roc_data(roc_d)
best = ordered_ROC_data[0]

# Display best threshold with its parameters
print(f"Best Threshold is {best[0]}")
print(f"\t\t with TP rate of {best[2]*100:>0.5f} %")
print(f"\t\t with FP rate of {best[3]*100:>0.5f} %")

# Reorders roc_data for plotting
def order_roc_data_for_plot(ordered_ROC_data):
    def sort_f(list_obj):
        return(list_obj[3])
    ordered_ROC_data.sort(key = sort_f)
    return ordered_ROC_data
order_roc_data_for_plot(ordered_ROC_data)

# Caluculates are under points
### Calculates the area under a sequence of points. 
def AUC(x_data,y_data):
    x = x_data.copy()
    y = y_data.copy()

    # Ensure it reaches certain points
    x.insert(0,0)
    y.insert(0,0)
    x.append(1)
    y.append(1)

    # Calculate area under points
    auc = 0
    prev = (0,0)
    for x_coord, y_coord in zip(x,y):
        auc += ((x_coord - prev[0]) * (y_coord - prev[1])) / 2
        auc += (x_coord - prev[0]) * (prev[1])

        prev = (x_coord, y_coord)
    return auc

# Plots the roc curve
def plot_roc_curve():
    l2 = order_roc_data_for_plot(ordered_ROC_data)
    x = [dictionary[3] for dictionary in l2]
    x.append(1)
    y = [dictionary[2] for dictionary in l2]
    y.append(1)
    j = np.arange(0, 1.1)

    plt.plot(x, y) 
    plt.plot(j, j, 'r--')
    plt.title("Roc Curve Classifier Model")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.figtext(0.3, 0.00, f"AUC = {AUC(x,y)}", wrap=True, horizontalalignment='center', fontsize=10)
    plt.show()

plot_roc_curve

# Claculates and displays the confusion matrix of the 
# a given threshold.
def Display_Confusion_Matrix(model, ids, annotations_mod, threshold):
    TP, FP, FN, TN = test_acc(model,ids, annotations_mod, threshold = threshold)
    
    full_conf = np.array([[TP, FP], [FN, TN]])
    labels = ["Infected", "Healthy"]

    fig, ax = plt.subplots(ncols = 1, figsize = (12, 12))

    disp = ConfusionMatrixDisplay(full_conf, display_labels = labels)
    disp.plot(ax=ax)
    disp.ax_.set_title("Confusion Matrix of the Classifier Model", fontdict = {"fontsize" : 25}, pad = 1)

    fig.savefig("Classifier_Model_Confusion_Matrix.png")
    plt.show()

Display_Confusion_Matrix(model, ids, annotations_mod, best[0])

# Make batches to calculate the general accuracy of the model
# according to the hold out data
def average_acc(model, ids, annotations_mod, threshold):
    Correct, Incorrect = 0, 0
    batches = 5
    batch_every = annotations_mod.shape[0] // batches
    n = 1

    batches_acc = []
    
    patient_data_iterator = get_patient_data(ids, annotations_mod=annotations_mod)
    for X, y in patient_data_iterator:
        if n == batch_every:
            batches_acc.append(Correct/(Correct+Incorrect))
            Correct, Incorrect = 0, 0
            n = 1
        n += 1
        
        pred_ratios = [0,0]# healthy, infected

        # Make prediction
        predictions = model(X)

        # Calculate ratio
        for pred in predictions:
            if pred[0] < pred[1]: # if predicted as infected
                pred_ratios[1] += 1 # add to infected
            else: 
                pred_ratios[0] += 1 # add to healthy
        ratio = pred_ratios[1] / sum(pred_ratios) # Infected ratio

        # Classify result
        if ratio > threshold:
            # Prediction is "infected"
            if y == 1: # y is infected
                Correct += 1 
            else: # y is healthy
                Incorrect += 1 
        else:
            # Prediction is "healthy"
            if y == 1: # y is infected
                Incorrect += 1 
            else: # y is healthy
                Correct += 1 

    return batches_acc

batches_acc = average_acc(model, ids, annotations_mod, best[0])

print(f"The average accuracy for the model is:")
print(f"\t{sum(batches_acc)/len(batches_acc)}\n")

print(f"The individual batches results are:")
print(f"\t{batches_acc}")
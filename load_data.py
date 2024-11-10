# This file contains functions and classes uses in other files to avoid 
# rebundancy and have a clearer code
import cv2
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
import os


### Get Both Data Files
def open_data(data_path):
    annotations = pd.read_csv(data_path+"PatientDiagnosis.csv")
    patch_data = pd.read_excel(data_path+"HP_WSI-CoordAllAnnotatedPatches.xlsx")
    return (annotations, patch_data)

### Obtain ids from the patient patches names
def get_ids(image_path = "Data\Annotated"):
    ids = ([name for name in os.listdir(image_path) ])
    ids_mod = [name[:-2] for name in ids]
    return ids, ids_mod

### Remove patches that we have no info on ()
def clean_patches(patch_data, ids_mod):
    patch_data = patch_data.drop(["i", "j", "h", "w"], axis = 1)
    patches_cleaned = patch_data.copy()

    for x in range (patch_data.shape[0] - 1, -1, -1):
        if patches_cleaned["Pat_ID"][x] not in ids_mod:
            patches_cleaned = patches_cleaned.drop(x)

    return patches_cleaned

### Get a dictionary to visualize the distribution of values in the dataset
def data_partitions(cleaned_patches):
    val = {}

    for x in range (patch_data.shape[0] - 1, -1, -1):
        if cleaned_patches["Presence"][x] not in val.keys():
            val[cleaned_patches["Presence"][x]] = 0
        val[cleaned_patches["Presence"][x]] += 1
    
    return val

### Load images
def load_images(cleaned_patches, data_path = "Data\Annotated\\"):
    all_images = []

    for x in range (cleaned_patches.shape[0] - 1, -1, -1):
        img_path = cleaned_patches["Pat_ID"][x]+"_"+(str(cleaned_patches["Section_ID"][x]))+"\\"#+str(cleaned_patches["Window_ID"][x])
        img_id = "0"*(5-len(str(cleaned_patches["Window_ID"][x])))+str(cleaned_patches["Window_ID"][x])+".png"

        if os.path.exists(data_path+img_path+img_id):
            img = torch.transpose(torch.from_numpy(cv2.imread(data_path+img_path+img_id)).to(torch.float), 0, 2)
            if patch_data["Presence"][x] == 1:
                diagnosis = (torch.tensor((0, 1))).to(torch.float) # Infected
            else:
                diagnosis = (torch.tensor((1, 0))).to(torch.float) # Healthy
            all_images.append({"img": img, "diagnosis" : diagnosis})
   
    return all_images

### A class that can serve as the dataset
class Cropped_Patches(Dataset):
    def __init__(self, img):
        self.img_data = img
    
    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, index):
        img = self.img_data[index]["img"]
        diagnosis = self.img_data[index]["diagnosis"]
        return [img, diagnosis]

### A stochastic Data Spliter
def st_data_splitter(all_images, chunks): # This is not very deterministic but will do for the moment
    random.shuffle(all_images)
    #print(all_images)
    partition = len(all_images)//chunks

    test_set = all_images[0: partition]
    train_set = all_images[partition : len(all_images)]
    return train_set, test_set 

### A non-stochastic Data Splitter
def data_splitter(all_images, n_chunks, choosen_chunk):
    if n_chunks == 1:
        return all_images

    size = len(all_images)
    chunks_dist = []
    for x in range(0, size, size//n_chunks):
        chunks_dist.append(x)

    chunks = []
    for x in range(0, len(chunks_dist)-1, 1):
        chunks.append(all_images[chunks_dist[x]:chunks_dist[x+1]])
    
    test_set = chunks[choosen_chunk]
    del chunks[choosen_chunk]
    train_set = []
    for chunk in chunks:
        train_set += chunk

    return train_set, test_set



### Main --------------------------------------------------------------- ###

# Example code for the functions above

data_path = "Data\\"
annotations, patch_data = open_data(data_path)

#ids, ids_mod = get_ids(image_path="Data\Annotated") 
#cleaned_patches = clean_patches(patch_data, ids_mod)
#partitions = data_partitions(cleaned_patches)
#all_images = load_images(cleaned_patches) 
#train_set, test_set = st_data_splitter(all_images, 5)
#train_set1, test_set = data_splitter(all_images, 5, 0)

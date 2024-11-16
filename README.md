# Medical-Imaging
This is a project from the Computer Vision class at UAB University.

## Task Description
The objective for this task is to classify patches of medical images to detect
the presence or lack of Helicobacter pylori. Thus our images are classified as
either infected or healthy.

We mainly focus on various methods of classification of individual patches
instead of making diagnosis for patients. 

We have implemented several models to compare their efficiency and accuracy.

## Autoencoder Model
[text here]


## Classifier
A simple convolutional neural network. 
It makes use of three convolutional layers and a linear layer at the end. 
It is complemented with several pooling layers to simplify the result.
It uses ReLU as the activation function.

With the dataset provided it has a 88% accuracy.

## "A ojo" Model
A simple algorithm that reads the image pixels row by row and if a severe difference
is detected between a pixel and the next it is classified as infected. This difference
is defined as a modifiable threshold.

At its best threshold value it holds a 88% accuracy.

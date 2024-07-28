# satellite-pose-estimation-infrared

This project is the main location of the Machine Learning code I wrote for my master's thesis. The code contains the architecture for a neural network that I used for training, as well as scripts for dealing with the dataset, such as loading the images, applying transformations, and handling the ground truth data effectively.

The code consists of several subdirectories for manipulating the pose dataset generated for the infrared satellite pose estimation research. These are summarized below.

## Dataset Manipulation

Tools to manipulate the dataset and understand the dataset size, ensure that each image has a corresponding pose associated with it.

## Image Equalization

Functions for converting the `.tiff` images from the thermal camera into equalized `.png` code for feeding into the machine learning model. It uses different equalization techniques based on OpenCV functions.

## Object Detection

Performs the first step, the object detection of the 2-stage machine learning model.

## Pose Estimation

A neural network for performing pose estimation on the resulting images.
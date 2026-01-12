# BrainTumorDetector

## Project Overview
This project implements an end-to-end deep learning pipeline for multi-class brain tumor classification using MRI scan images. A custom Convolutional Neural Network (CNN) was developed in PyTorch to classify brain MRI images into four clinically relevant categories:
Healthy
Pituitary Tumor
Glioma
Meningioma
The goal of this project is to demonstrate applied skills in computer vision, medical imaging, deep learning model development, and evaluation.

## Dataset
https://www.kaggle.com/datasets/miadul/brain-tumor-mri-dataset/data

## Model Architecture
A custom CNN was implemented with the following structure:
3 Convolutional Blocks
Convolution → Batch Normalization → ReLU → Max Pooling
Fully Connected Classifier
Dense layer (256 units)
Dropout (0.5)
Output layer (4 classes)
Input images are resized to 224×224, converted to grayscale, and normalized using dataset-specific mean and standard deviation.

## Model Performance
Test Accuracy: 94.2%
Validation Accuracy: ~95%

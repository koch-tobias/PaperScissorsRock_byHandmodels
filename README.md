# PaperScissorsRock_byHandmodels
This project contains the source code and scientific work for the master project Explainable Machine Learning (xAI-Proj-M) at the University of Bamberg. 

The goal of the project is to develop a machine learning application considering the three steps Data Engineering, ML Model Engineering and ML Model Evaluation, which are part of the Deep Learning Life Cycle.
A separate research question is processed for each step but our overall goal is to detect handsignes based on images.

<p align="center">
  <img width="300" src="images/CRISP-ML.png">
</p>

## How to set up 
Firstly you have to clone the repository on your local device. Afterwards you can create the predefined environment by using "conda enc create -f environment.yml".

Now the set up is ready to use and you can run the file "main.py". 

## Dataset
For our project we combined the following three subdataset with in total 7477 images:
- [Roboflow Dataset](https://public.roboflow.com/classification/rock-paper-scissors) with 2925 images
  <p float="right">
    <img src="data_original/dataset_1/train/rock/rock01-000_png.rf.560ebe5b8570f6866c33946448ccf7de.jpg" width="150" />
    <img src="data_original/dataset_1/train/paper/paper01-000_png.rf.02152baa06324655efacad9c5bda9f1a.jpg" width="150" /> 
    <img src="data_original/dataset_1/train/scissors/scissors01-000_png.rf.bc8ea3d7b607fa5306391e214675bc07.jpg" width="150" /> 
  </p>

- [Kaggle Dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) with 2188 images
  <p float="right">
    <img src="data_original/dataset_2/rock/0bioBZYFCXqJIulm.png" width="150" />
    <img src="data_original/dataset_2/paper/0a3UtNzl5Ll3sq8K.png" width="150" /> 
    <img src="data_original/dataset_2/scissors/0CSaM2vL2cWX6Cay.png" width="150" /> 
  </p>

- [Kaggle Dataset](https://www.kaggle.com/datasets/glushko/rock-paper-scissors-dataset) with 2364 images
  <p float="right">
    <img src="data_original/dataset_3/train/rock/glu_235.png" width="150" />
    <img src="data_original/dataset_3/train/paper/glu_161.png" width="150" /> 
    <img src="data_original/dataset_3/train/scissors/glu_116.png" width="150" /> 
  </p>

## Data Engineering

## ML-Model Engineering
In this part we want to consider the research question "What impact does a pre-trained network (alexnet) have on performance compared to a simple CNN model?". 

## Model Evaluation

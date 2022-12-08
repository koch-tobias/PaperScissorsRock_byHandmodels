# PaperScissorsRock_byHandmodels
This project contains the source code and scientific work for the master project "Explainable Machine Learning (xAI-Proj-M)" at the University of Bamberg. 

The goal of the project is to develop a machine learning application considering the three steps data engineering, ML model engineering and ML model evaluation, which are part of the deep learning life cycle.
A separate research question is processed for each step but our overall goal is to detect the hand signs (paper, rock, scissors) based on images.

<p align="center">
  <img width="300" src="images/CRISP-ML.png">
</p>

## Set up 
Clone the repository on your local device.

### Local Execution

#### Virtual Environment

##### Prerequisites local

- You have [Python](https://www.python.org/) installed on your machine
- You have [Anaconda](https://www.anaconda.com/) or at least [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your machine

##### Install your environment

Open a PowerShell/Terminal window, navigate to your project folder and setup your environment via `conda env create -f environment.yml`. To update your existing environment use `conda env update`

##### Execute your code local

Enter following command in your terminal while you are in your project root directory

```terminal
python src/main.py
```

The default setting of main.py is to load and evaluate the best model (a transfer learning model) which is based on the efficientNet_B0 model provided by pyTorch.
It also outputs predictions for random images from the validation dataset.

You can change this setting in the main.py file.

These following options are available:
  1. Train new transfer learning model: </br>
    a) Set the boole variable "train_new_transferlearning_model" = True </br>
    b) Set the hyperparameter values you want to use </br>
    c) Save the changes </br>
    d) Run the main.py file again </br>
  2. Evaluate an existing model: </br>
    a) Change the path for the variable "model_folder" to your model you want to evaluate </br>
    b) Set the boole variable "evaluate_existing_model" = True </br>
    c) Set the number of images you want to evaluate in variable "num_images_val" </br>
    d) Save the changes </br>
    e) Run the main.py file again </br>
  3. Make prediction on a single image: </br>
    a) Change the path for the variable "single_image_path" to your image you want to classify </br>
    b) Set the boole variable "prediction_on_single_image" = True </br>
    c) Save the changes </br>
    d) Run the main.py file again </br>


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
- [giusti2018](https://github.com/alessandro-giusti/rock-paper-scissors) with 2372 images

## Data Engineering
In this part we want to consider the research question "Which data augmentation methods can increase the model's performance in classifying images of „Rock, Paper, Scissors“ hand signs?"
## ML-Model Engineering
In this part we want to consider the research question "Can transfer learning bring a benefit on the performance of CNN models for Rock, Paper, Scissors?". 

## Model Evaluation

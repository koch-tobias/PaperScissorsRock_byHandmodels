# %% [markdown]
# # **Data engineering**

# %% [markdown]
#  Research question: "Which data augmentation and data formatting methods could be useful for classifying images of "Rock, Paper, Scissors“ hand signs?"

# %% [markdown]
# ## Imports
import os
#import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import torch
import torchvision.transforms as T
from PIL import Image
#from torch import nn
#import pandas as pd
import os
import shutil
import splitfolders
#sys.path.append('/Users/satyamapantagomeza/DataspellProjects/PaperScissorsRock_byHandmodels')

# %% [markdown]
# ## Function to load the datasets
def loading(folder_name: str):
    direc = os.path.join(
        folder_name)
    return os.listdir(direc), len(os.listdir(direc))

# %% [markdown]
# ## Function to print example images for the dataset

# %%
def plot_image_grid(images, labels, subfolder):
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

    f, axarr = plt.subplots(len(subfolder),len(labels))
    plt.rc('font', **font)

    count_img = 0
    for i in range(len(subfolder)):
        for n in range(len(labels)):
            if len(subfolder) == 1:
                axarr[n].imshow(images[count_img])
                axarr[n].grid(False)     
                axarr[n].set_xticks([])
                axarr[n].set_yticks([])
                if count_img < len(labels):
                    axarr[n].set_title(labels[n],**font)
                count_img = count_img + 1     
            else:
                axarr[i,n].imshow(images[count_img])
                axarr[i,n].grid(False)     
                axarr[i,n].set_xticks([])
                axarr[i,n].set_yticks([])
                if n == 0:
                    axarr[i,n].set_ylabel(subfolder[i], loc='center', rotation=0, labelpad = 20, **font)
                if count_img < len(labels):
                    axarr[i,n].set_title(labels[n], **font)
                count_img = count_img + 1

# %% [markdown]
# ## Function to get the number of subfolder in a folder

# %%
def count_subfolder(folder_path: str):
    startinglevel = folder_path.count(os.sep)
    num_subfolders = []
    for top, dirs, files in os.walk(folder_path):
        level = top.count(os.sep) - startinglevel
        num_subfolders.append(level)
    return max(num_subfolders)

# %% [markdown]
## Analyzing the datasets

# %%
def analyze_dataset(dataset_dir):
    subfolder = ['test','train','val']
    subfolder_labels = ['rock','paper','scissors']
    
 
    datasets, num_datasets = loading(dataset_dir)

    if '.DS_Store' in datasets:
        print("Analyzing " + str(num_datasets-1) + " datasets...")
    else:
        print("Analyzing " + str(num_datasets) + " datasets...")

    
    for dataset in datasets:
        totel_num_images = 0
        temp_images = []
        
        if dataset == '.DS_Store':
            continue
        elif count_subfolder(folder_path=(dataset_dir + "/" + dataset)) <= 1:
            for label in subfolder_labels:
                count = 0
                path = dataset_dir + "/" + dataset + "/" + label
                images, num_images = loading(path)
                totel_num_images = totel_num_images + num_images

                for img in images:
                    if (img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")) and count < 1:
                        img = Image.open(path + "/" + img)
                        temp_images.append(img)
                        count = count + 1
            plot_image_grid(temp_images, subfolder_labels,subfolder=['no split'])


            print("Total number of images in " + dataset + ": " + str(totel_num_images))

        else:
            for folder in subfolder:
                for label in subfolder_labels:
                    count = 0
                    path = dataset_dir + "/" + dataset + "/" + folder + "/" + label
                    images, num_images = loading(path)
                    totel_num_images = totel_num_images + num_images

                    for img in images:
                        if (img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")) and count < 1:
                            img = Image.open(path + "/" + img)
                            temp_images.append(img)
                            count = count + 1
            plot_image_grid(temp_images, subfolder_labels,subfolder)


            print("Total number of images in " + dataset + ": " + str(totel_num_images))

# %% [markdown]
# ## Combining the datasets and saving in a new folder

# %% 
def add_new_dataset(dataset_path: str):
    split = ['test','train','val']
    labels = ['rock','paper','scissors']
    target_path = '../data_combined/dataset_witout_split/'
     
    subfolders, num_subfolders = loading(dataset_path)
    dir_depth = count_subfolder(folder_path=dataset_path)

    print("Add dataset to the combined dataset..")

    for subf in subfolders:
        if dir_depth == 1 and subf in labels:
            path = dataset_path + "/" + subf
            dst = target_path + subf
            images, num_images = loading(path)
            for img in images:
                src = path + "/" + img
                shutil.copy(src, dst)
        elif subf in split:
            path_splits = dataset_path + "/" + subf
            label_folders, num_label_folders = loading(path_splits)
            for label in label_folders:
                if label == ".DS_Store":
                    continue
                else:
                    path_labels = path_splits + "/" + label
                    dst = target_path + label
                    images, num_images = loading(path_labels)
                    for img in images:
                        src = path_labels + "/" + img
                        shutil.copy(src, dst)

    print("Dataset successfully added!")

# %% [markdown]
# ## Split dataset
def split():
    src = '../data_combined/dataset_without_split'
    dst = '../data_combined/dataset_splitted'
    splitfolders.ratio(src, output=dst, seed=1337, ratio=(.7, 0.1,0.2)) 


# %% [markdown]
# ## To do: Function to transform images to same size
def transform_img():
    subfolder = ['rock','paper','scissors']
    for i in subfolder:
        x = loading(i)
        for img in x:
            if (img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")):
                img = Image.open(r"/Users/satyamapantagomeza/DataspellProjects/PaperScissorsRock_byHandmodels/data_original/Combined_dataset/"+i+"/"+img)

                width, height = img.size
                left = 4
                top = height / 5
                right = 154
                bottom = 3 * height / 5
                im1 = img.crop((left, top, right, bottom))
                newsize = (300, 300)
                im1 = im1.resize(newsize)
        print("reshaped {0} image size: {1}".format(i,im1.size))
transform_img()

def transform_img():
    subfolder = ['rock', 'paper', 'scissors']
    for i in subfolder:
        x = loading(i)
        for img in x:
            img = mpimg.imread("/Users/satyamapantagomeza/DataspellProjects/PaperScissorsRock_byHandmodels/data_original/Combined_dataset/"+i+"/"+img)
            # np.asarray(img).shape
            resized_imgs = [T.Resize(size=size)(img) for size in [32,128]]
            plot(resized_imgs,col_title=["32x32","128x128"])


# %% [markdown]
# ## To do: Function for Data preprocessing 

# %%

# %% [markdown]
# ## To do: Function for Data Augmentation (Shifting, flipping, changing brightness, rotation, adding noise,..) 

# %%


# %% [markdown]
# ## To do: Function for Data Augmentation (Shifting, flipping, changing brightness, rotation, adding noise,..) 

# %% [markdown]
# ## Main-method

# %%
# Add new dataset to the combined dataset
add_dataset = False
dir_new_dataset = '../data_original/dataset_X'

if add_dataset == True:
    add_new_dataset(dataset_path=dir_new_dataset)
else: 
    print("Datasets are already combined!")

# %%
# Split data into train, validation and testset 
split_dataset = False

if split_dataset == True:
    split()
else: 
    print("Datasets is already splitted!")

# %%
# Analyzing the original datasets
analyze_dataset("../data_original")

# %%
# Analyzing the combined dataset
analyze_dataset("../data_combined")

# %%
# Transform the combined dataset
transform_img()




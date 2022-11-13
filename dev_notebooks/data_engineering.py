# %% [markdown]
# # **Data engineering**

# %% [markdown]
#  Research question: "Which data augmentation and data formatting methods could be useful for classifying images of "Rock, Paper, Scissors“ hand signs?"

# %% [markdown]
# ## Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
#from torch import nn
import pandas as pd
import os
import kornia
import shutil
import splitfolders
from torchvision.transforms import transforms
#from skimage.util import random_noise

# %% [markdown]
# ## Function to list and count the number of dircetories inside a folder

# %%
def loading(folder_name: str):
    direc = os.path.join(
        folder_name)
    return os.listdir(direc), len(os.listdir(direc))

# %% [markdown]
# ## Function to print example images for the dataset

# %%
def plot_image_grid(images, labels, subfolder,dataset):
    font = {'family' : 'sans-serif',
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
    plt.savefig('../images/'+dataset)

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
# ## Analyzing the datasets

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
            plot_image_grid(temp_images, subfolder_labels,subfolder=['no split'],dataset=dataset)
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
            plot_image_grid(temp_images, subfolder_labels,subfolder,dataset)


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
## Split dataset

# %%
def split():
    src = '../data_combined/dataset_without_split'
    dst = '../data_combined/dataset_splitted'
    splitfolders.ratio(src, output=dst, seed=1337, ratio=(.7, 0.1,0.2))


# %% [markdown]
# ## Function to transform images to same size

# %%
def transform_img(img_path: str):
    img = Image.open(img_path)
    newsize = (300, 300)
    img_resize = img.resize(newsize)

    return img_resize

# %% [markdown]
# ## Function to transform images from RGBA to RGB

# %%
def rgba_to_rgb(dir_dataset="../data_combined/dataset_splitted"):
    #Convert all rgba images as rbg images and replace it in the dataset
    split = ['test','train','val']
    labels = ['rock','paper','scissors']
    folders, num_folders = loading(dir_dataset)

    for folder in folders:
        if folder in split:
            dir_folder = dir_dataset + "/" + folder
            subfolders, num_subfolders = loading(dir_folder)
            for subfolder in subfolders:
                if subfolder in labels:
                    dir_subfolder = dir_folder + "/" + subfolder
                    images, num_images = loading(dir_subfolder)
                    for image in images:
                        dir_images = dir_subfolder + "/" + image
                        with Image.open(dir_images) as img:
                            if img.mode == "RGBA":
                                img_rgb = img.convert("RGB")
                                img_rgb.save(dir_images)
                            else:
                                continue
        elif folder in labels:
            dir_folder = dir_dataset + "/" + folder
            images, num_images = loading(dir_folder)
            for image in images:
                dir_images = dir_subfolder + "/" + image
                with Image.open(dir_images) as img:
                    if img.mode == "RGBA":
                        img_rgb = img.convert("RGB")
                        img_rgb.save(dir_images)
        else:
            continue

# %% [markdown]
# ## Functions for analyzing selected Data Augmentation techniques 

# %%
def shift_image(image_path:str): 
    #shift image in different areas like forward, backward ,top
    img = np.array(Image.open(image_path))

    img_tensor = torch.tensor(img.transpose([2, 0, 1])).float()

    affine = kornia.augmentation.RandomAffine(degrees=0, translate=(0.3, 0.3), padding_mode='border')
    for i in range(3):
        for j in range(3):
            ax = plt.subplot(4, 4, i*4 + j +1)
            img_translated = affine(img_tensor)
            ax.imshow(img_translated.squeeze().permute(1, 2, 0).byte())
    plt.show()

# %%
def horizontal_flip(image_path:str):
    img = Image.open(image_path)
    for i in range(2):
        for j in range(1):
            ax = plt.subplot(4, 4, i*4 + j +1)
            transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.9)])
            img=transform(img)
            ax.imshow(img)
    plt.show()

# %%
def random_rotation(image_path:str):
    img = Image.open(image_path)
    for i in range(3):
        for j in range(3):
            ax = plt.subplot(4, 4, i*4 + j +1)
            transform = T.RandomRotation(degrees=(60, 90))
            img=transform(img)
            ax.imshow(img)
    plt.show()
# %%
def gausian_blur(image_path:str):
    img = Image.open(image_path)
    for i in range(3):
        for j in range(3):
            ax = plt.subplot(4, 4, i*4 + j +1)
            transform = T.GaussianBlur(kernel_size=(7, 13), sigma=(9, 9))
            img=transform(img)
            ax.imshow(img)
    plt.show()
# %%
def random_crop(image_path:str):
    img = Image.open(image_path)
    for i in range(3):
        for j in range(3):
            ax = plt.subplot(4, 4, i*4 + j +1)
            transform = T.RandomCrop((250,300), padding=50)
            img=transform(img)
            ax.imshow(img)
    plt.show()

# %%
def noise(image_path:str, sigma:float):
    img = Image.open(image_path)
    for i in range(3):
        for j in range(3):
            ax = plt.subplot(4, 4, i*4 + j +1)
            img = random_noise(img, sigma**2)
            ax.imshow(img)
    plt.show()

# %%
def plot_data_augmentation(image_path:str):
    print("Image after random crop:")
    random_crop(image_path)
    print("Image after gausian:")
    gausian_blur(image_path)
    print("Image after rotation:")
    random_rotation(image_path)
    print("Image after horizontal flip:")
    horizontal_flip(image_path)
    print("Image after shifting:")
    shift_image(image_path)
    print("Image after adding noise:")
    noise(image_path,sigma=0.177)

# %%
plot_data_augmentation(image_path="../data_combined/dataset_without_split/paper/nasmi_198.png")

# %% [markdown]
# ## Function to transform each image in the dataset so same size and apply selected data augmentation techniques

# %%
def manual_transformation(dir_dataset:str, img_crop=False, img_gausian=False,img_rotation=False, img_hflip=False, img_shift=False):
    train_dataset = torchvision.datasets.ImageFolder(root=dir_dataset + "/train")
    val_dataset = torchvision.datasets.ImageFolder(root=dir_dataset + "/val")

    train_x = []
    train_y = []
    val_x = []
    val_y = []
    
    for img in train_dataset:
        newsize = (300, 300)
        img_resize = img[0].resize(newsize)
        train_x.append(np.array(img_resize))
        train_y.append(img[1])
        if img_crop == True:
            transform = T.RandomCrop((250,300), padding=50)
            img_new=transform(img_resize)
            train_x.append(img_new)
            train_y.append(img[1])
        elif img_gausian == True:
            transform = T.GaussianBlur(kernel_size=(7, 13), sigma=(9, 9))
            img_new=transform(img_resize)
            train_x.append(img_new)
            train_y.append(img[1])
        elif img_rotation == True:
            transform = T.RandomRotation(degrees=(60, 90))
            img_new=transform(img_resize)
            train_x.append(img_new)
            train_y.append(img[1])
        elif img_hflip == True:
            transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.9)])
            img_new=transform(img_resize)
            train_x.append(img_new)
            train_y.append(img[1])
        else:
            continue
    
    for img in val_dataset:
        newsize = (300, 300)
        img_resize = img[0].resize(newsize)
        val_x.append(img_resize)
        val_y.append(img[1])
        break

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)

    return train_x, val_x, train_y, val_y


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
# Transform RGBA images to RGB Images
rgba_to_rgb(dir_dataset="../data_combined/dataset_splitted")

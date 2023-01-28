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
from skimage.util import random_noise
from tqdm import tqdm
import random
import albumentations as A
#import cv2

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
    plt.savefig('../images/datasets/'+ dataset + '.png')

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
        elif count_subfolder(folder_path=(dataset_dir)) == 2:
            for label in subfolder_labels:
                count = 0
                path = dataset_dir + "/" + dataset + "/" + label
                images, num_images = loading(path)
                totel_num_images = totel_num_images + num_images
                rand = random.randrange(0, len(images))

                for img in images:
                    if count == rand:
                        img = Image.open(path + "/" + img)
                        temp_images.append(img)
                    count = count + 1
            plot_image_grid(temp_images, subfolder_labels,subfolder=['no split'],dataset=dataset)
            print("Total number of images in " + dataset + ": " + str(totel_num_images))
        elif count_subfolder(folder_path=(dataset_dir)) == 1:
            count = 0
            path = dataset_dir + "/" + dataset
            images, num_images = loading(path)
            totel_num_images = totel_num_images + num_images
            rand = random.randrange(0, len(images))
            for img in images:
                if count == rand:
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
                    rand = random.randrange(0, len(images))
                    for img in images:
                        if count == rand:
                            img = Image.open(path + "/" + img)
                            temp_images.append(img)
                        count = count + 1
            plot_image_grid(temp_images, subfolder_labels,subfolder,dataset)
            print("Total number of images in " + dataset + ": " + str(totel_num_images))

# %% [markdown]
# ## Analyzing the evaluation datasets
def analyze_eval_dataset(dataset_dir):
    images, num_images = loading(dataset_dir)
    temp_images = []
    num_samples = 16
    rand = random.sample(range(0, len(images)), num_samples)
    for i in range(num_samples):
        count = 0
        for img in images:
            #print(str(count) + " == " + str(rand))
            if count == rand[i]:
                img = Image.open(dataset_dir + "/" + img)
                temp_images.append(img)
            count = count + 1

    # Plot the images
    f, axarr = plt.subplots(4,4)
    count_img = 0
    #print(temp_images)
    for i in range(4):
        for n in range(4):
            axarr[i,n].imshow(temp_images[count_img])
            axarr[i,n].grid(False)
            axarr[i,n].set_xticks([])
            axarr[i,n].set_yticks([])
            count_img = count_img + 1

    plt.savefig("../images/datasets/sampels_evaldataset.png")
    print("Total number of images in the eval dataset: " + str(num_images))
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
def split(original_dataset_dir: str,seed: int):
    #Truncate the existing combined dataset folder
    dst_dir = 'data_combined'
    splits, num_splits = loading(dst_dir)
    for split in splits:
        if split == 'train' or split == 'val':
            split_dir = dst_dir + '/' + split
            shutil.rmtree(split_dir)
    
    #Split original datasets into train&validation (80/20) and store it as combined dataset
    datasets, num_datasets = loading(original_dataset_dir)
    for dataset in datasets:
        src = original_dataset_dir + '/' + dataset
        dst = 'data_combined'
        splitfolders.ratio(src, output=dst, seed=seed, ratio=(0.8, 0.2))

    rgba_to_rgb()

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
def rgba_to_rgb(dir_dataset="C:/Users/TKC/Desktop/Uni/PaperScissorsRock_byHandmodels/data_original/dataset_5"):
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
    plt.savefig("../images/data_augmentation/example_shift.png")
    plt.show()

# %%
def horizontal_flip(image_path:str):
    img = Image.open(image_path)
    for i in range(2):
        for j in range(1):
            ax = plt.subplot(4, 4, i*4 + j +1)
            transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
            img=transform(img)
            ax.imshow(img)
    plt.savefig("../images/data_augmentation/example_hflip.png")
    plt.show()

# %%
def random_rotation(image_path:str):
    img = Image.open(image_path)
    for i in range(3):
        for j in range(3):
            ax = plt.subplot(4, 4, i*4 + j +1)
            transform = T.RandomRotation(degrees=(15))
            img=transform(img)
            ax.imshow(img)
    plt.savefig("../images/data_augmentation/example_rotation.png")
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
    plt.savefig("../images/data_augmentation/example_gaus.png")
    plt.show()
# %%


# %%
def noise(image_path:str):
    img = Image.open(image_path)
    im_arr = np.asarray(img)
    noise_img = random_noise(im_arr, mode='poisson', seed=42, clip=False)
    noise_img = (255*noise_img).astype(np.uint8)
    image = Image.fromarray(noise_img)

    plt.imshow(image)
    plt.savefig("../images/data_augmentation/example_noise.png")
    plt.show()

def spatial_distortion(image_path:str):
    img = Image.open(image_path)
    im_arr = np.asarray(img)
    transform = A.Compose([
    A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.1),
    A.OpticalDistortion(distort_limit=0.2, shift_limit=0.05, p=0.1)
    ])
    plt.imshow(transform(image=im_arr)['image'])
    plt.savefig("../images/data_augmentation/example_dist.png")
    plt.show()

def colour_jitter(image_path:str):
    img = Image.open(image_path)
    for i in range(3):
        for j in range(3):
            ax = plt.subplot(4, 4, i*4 + j +1)
            transform = transforms.ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1)
            ax.imshow(img)
    plt.savefig("../images/data_augmentation/colour_jitter.png")
    plt.show()

def random_affine(image_path:str):
    img = Image.open(image_path)
    for i in range(3):
        for j in range(3):
            ax = plt.subplot(4, 4, i*4 + j +1)
            transform = transforms.RandomAffine(degrees=(30, 70),
            translate=(0.1, 0.3), scale=(0.5, 0.75))
            ax.imshow(img)
    plt.savefig("../images/data_augmentation/random_affine.png")
    plt.show()
    
def padded_image(image_path:str):
    img = Image.open(image_path)
    for i in range(3):
        for j in range(3):
            ax = plt.subplot(4, 4, i*4 + j +1)
            transform = transforms.Pad((10, 20, 50, 50))
            ax.imshow(img)
    plt.savefig("../images/data_augmentation/random_rgb_shift.png")
    plt.show()



# In Albumentations, this interface is available as A.Compose() which lets us define the augmentation pipeline with the list of augmentations we want to use

#applying combination of data augmentation techniques: RandomCrop,HorizontalFlip,RandomBrightnessContrast
def comb_transformation_1(image_path:str):
    img=Image.open(image_path)
    im_arr=np.array(img)
    transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ])

    plt.imshow(transform(image=im_arr)['image'])
    plt.savefig("../images/data_augmentation/example_comb1.png")
    plt.show()

#applying combination of data augmentation techniques: RandomSnow,HueSaturation,ChannelShuffle
def comb_transformation_2(image_path:str):
    img=Image.open(image_path)
    im_arr=np.array(img)
    transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,rotate_limit=15,p=0.5),
    A.Superpixels(p_replace=0.1, n_segments=10, max_size=128, interpolation=1, always_apply=False, p=0.5),
    A.ChannelShuffle(p=1),
    ], p=1)

    plt.imshow(transform(image=im_arr)['image'])
    plt.savefig("../images/data_augmentation/example_comb2.png")
    plt.show()
#applying combination of data augmentation techniques: ShiftScaleRotate,ColorJitter,MotionBlur,CoarseDropout,ChannelDropout,GridDistortion,OpticalDistortion

def comb_transformation_3(image_path:str):
    img=Image.open(image_path)
    im_arr=np.array(img)
    transform = A.Compose([
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
    A.GaussNoise(var_limit=(0, 255), p=0.1),
    A.Blur (blur_limit=7, always_apply=False, p=0.5),
    A.CoarseDropout(max_holes=6, max_height=32, max_width=32, p=0.1),
    A.ChannelDropout(p=0.05)
    ])
    plt.imshow(transform(image=im_arr)['image'])
    plt.savefig("../images/data_augmentation/example_comb3.png")
    plt.show()

# %%
def plot_data_augmentation(image_path:str):
    # print("Image after random crop:")
    # random_crop(image_path)
    print("Image after gausian blur:")
    gausian_blur(image_path)
    print("Image after rotation:")
    random_rotation(image_path)
    print("Image after horizontal flip:")
    horizontal_flip(image_path)
    print("Image after shifting:")
    shift_image(image_path)
    print("Image after padding:")
    padded_image(image_path)
    print("Image after adding noise:")
    noise(image_path)
    print("Image after adding spatial distortion:")
    spatial_distortion(image_path)
    print("Image after colour jitter:")
    colour_jitter(image_path)
    print("Image after affine transform:")
    random_affine(image_path)
    print("Image after applying combination of augmentation techniques_1:")
    comb_transformation_1(image_path)
    print("Image after applying combination of augmentation techniques_2:")
    comb_transformation_2(image_path)
    print("Image after applying combination of augmentation techniques_3:")
    comb_transformation_3(image_path)

# %%
#plot_data_augmentation(image_path="/Users/satyamapantagomeza/Desktop/PaperScissorsRock_byHandmodels/data_original/dataset_1/paper/paper-hires1_png.rf.bf14bb5fd86e4d28a00897e40459f192.jpg")

# %% [markdown]
# ## Function to transform each image in the dataset so same size and apply selected data augmentation techniques
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
# %%
def manual_transformation_augmentation(dir_dataset:str, img_gausian=False,img_rotation=False, img_hflip=False, img_noise=False, img_shift=False,spat=False,comb_aug1=False,comb_aug2=False,comb_aug3=False,img_color_jitter=False,img_affine_transform=False,img_padding=False):
    train_dataset = torchvision.datasets.ImageFolder(root=dir_dataset + "/train")
    val_dataset = torchvision.datasets.ImageFolder(root=dir_dataset + "/val")

    train_x = []
    train_y = []
    val_x = []
    val_y = []

    for img in tqdm(train_dataset):
        newsize = (300, 300)
        img_resize = img[0].resize(newsize)
        train_x.append(np.asarray(img_resize))
        train_y.append(img[1])
        if img_gausian == True:
            transform = T.GaussianBlur(kernel_size=(7, 13), sigma=(9, 9))
            img_new=transform(img_resize)
            train_x.append(np.array(img_new))
            train_y.append(img[1])
        elif img_rotation == True:
            transform = T.RandomRotation(degrees=(60, 90))
            img_new=transform(img_resize)
            train_x.append(np.array(img_new))
            train_y.append(img[1])
        elif img_hflip == True:
            transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.9)])
            img_new=transform(img_resize)
            train_x.append(np.array(img_new))
            train_y.append(img[1])
        elif img_noise == True:
            im_arr = np.asarray(img_resize)
            noise_img = random_noise(im_arr, mode='poisson', seed=42, clip=False)
            noise_img = (255*noise_img).astype(np.uint8)
            img_new = Image.fromarray(noise_img)
            train_x.append(np.array(img_new))
            train_y.append(img[1])
        elif img_shift == True:
            im_arr = np.asarray(img_resize)
            img_tensor = torch.tensor(im_arr.transpose([2, 0, 1])).float()
            affine = kornia.augmentation.RandomAffine(degrees=0, translate=(0.3, 0.3), padding_mode='border')
            shift_img = affine(img_tensor)
            img_new = shift_img.squeeze().permute(1, 2, 0).byte()
            train_x.append(np.array(img_new))
            train_y.append(img[1])
        if img_color_jitter == True:
            transform = transforms.ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1)
            img_new=transform(img_resize)
            train_x.append(np.array(img_new))
            train_y.append(img[1])
        if img_affine_transform == True:
            transform = kornia.augmentation.RandomRGBShift(degrees=0, translate=(0.3, 0.3), padding_mode='border')  
            transform = transforms.RandomAffine(degrees=(15),scale=0.05)
            img_new=transform(img_resize)
            train_x.append(np.array(img_new))
            train_y.append(img[1])
        if img_padding == True:
            transform = transforms.Pad((10, 20, 50, 50))
            img_new=transform(img_resize)
            train_x.append(np.array(img_new))
            train_y.append(img[1])

        elif comb_aug1 == True:
            transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ])
            img_new=transform(img_resize)
            train_x.append(np.array(img_new))
            train_y.append(img[1])
        elif comb_aug2 == True:
            transform = A.Compose([A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                           rotate_limit=15,p=0.5),
                           A.Superpixels(p_replace=0.1, n_segments=100, max_size=128, interpolation=1, always_apply=False, p=0.5),
                        A.ChannelShuffle(p=1),#Randomly rearrange channels of the input RGB image.
            ], p=1)
            img_new=transform(img_resize)
            train_x.append(np.array(img_new))
            train_y.append(img[1])
        elif comb_aug3 == True:
            transform = A.Compose([
               A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
               A.GaussNoise(var_limit=(0, 255), p=0.1),
               A.Blur (blur_limit=7, always_apply=False, p=0.5),
               A.CoarseDropout(max_holes=6, max_height=32, max_width=32, p=0.1),
               A.ChannelDropout(p=0.05)
            ])
            img_new=transform(img_resize)
            train_x.append(np.array(img_new))
            train_y.append(img[1])
        # elif spat==True:
        #     transform = A.Compose([
        #     A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.1),
        #     A.OpticalDistortion(distort_limit=0.2, shift_limit=0.05, p=0.1)
        #     ])
        #     img_new=transform(img_resize)
        #     train_x.append(np.array(img_new))
        #     train_y.append(img[1])

        else:
            continue

    for img in val_dataset:
        newsize = (300, 300)
        img_resize = img[0].resize(newsize)
        val_x.append(np.array(img_resize))
        val_y.append(img[1])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)

    return train_x, val_x, train_y, val_y

# def manual_transformation(comb_1: bool, comb_2: bool, comb_3: bool):
#     if comb_1:
#         manual_transforms = transforms.Compose([
#                                 transforms.Resize((384,384)),
#                                 transforms.RandomRotation(degrees=(-15, 15)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                                 ])
#     elif comb_2:
#         manual_transforms = transforms.Compose([
#                                 transforms.Resize((384,384)),
#                                 transforms.RandomHorizontalFlip(p=0.4),                            
#                                 transforms.ToTensor(),
#                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                                 ])
#     elif comb_3:
#         manual_transforms = transforms.Compose([
#                                 transforms.Resize((384,384)),
#                                 transforms.ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1),
#                                 transforms.RandomAffine(degrees=(15),scale=0.05),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                                 ])
#     return manual_transforms
def manual_transformation(comb_1: bool, comb_2: bool,comb_3: bool,comb_4:bool,comb_5:bool,comb_6:bool,comb_7:bool):
    if comb_1:#only for rotation
        manual_transforms = transforms.Compose([
                                transforms.Resize((384,384)),
                                transforms.RandomRotation(degrees=(-15, 15)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])
    elif comb_2:#for rotation,translation,scaling and shearing (parameter values as per the paper)
        manual_transforms = transforms.Compose([
                            transforms.Resize((384,384)),
                            transforms.RandomAffine(degrees=(-15,15),translate=(0.15,0.3),scale=(0.85,1.15),shear=(0.85,1.15)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            ])
    elif comb_3:#for rotation,translation,scaling and shearing (with different parameter values as per the paper)
        manual_transforms = transforms.Compose([
                            transforms.Resize((384,384)),
                            transforms.RandomAffine(degrees=(-90,90),translate=(0.15,0.3),shear=(0.85,1.15)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            ])
    elif comb_4:#only for horizontal flipping
        manual_transforms = transforms.Compose([
                                transforms.Resize((384,384)),
                                transforms.RandomHorizontalFlip(p=0.4),                            
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])
    elif comb_5:#horizontal flipping with combination of rotation (parameter values as per the paper)
        manual_transforms = transforms.Compose([
                                transforms.Resize((384,384)),
                                transforms.RandomRotation((-30,30)),
                                transforms.RandomHorizontalFlip(p=0.5),#Default p value as per pytorch
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])
    elif comb_6:#only colour jitter
        manual_transforms = transforms.Compose([
                                transforms.Resize((384,384)),
                                transforms.ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ])
    
    elif comb_7:#colour jitter with rotation,translation and horizontal flip
        manual_transforms = transforms.Compose([
                        transforms.Resize((384,384)),
                        transforms.ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1),
                        transforms.RandomAffine(degrees=(-10,10)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ])
        
    return manual_transforms


#train_x, val_x, train_y, val_y = manual_transformation("../data_combined/dataset_splitted")

# %% [markdown]
# ## Main-method

# %%
# Analyzing the original datasets
#analyze_dataset("../data_original")

# %%
# Analyzing the combined dataset
#analyze_dataset("../data_combined")

# %%
# Analyzing the eval datasets
#analyze_eval_dataset("../data_own_images")

# %%
# Transform RGBA images to RGB Images
#rgba_to_rgb(dir_dataset="../data_combined/dataset_splitted")

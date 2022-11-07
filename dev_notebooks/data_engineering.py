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
# import torch
# import torchvision.transforms as T
from PIL import Image
from src import config
# from torch import nn
sys.path.append('/Users/satyamapantagomeza/DataspellProjects/PaperScissorsRock_byHandmodels')
import pandas as pd
# %% [markdown]
# ## To do: Function to load the datasets (Satyam)
import os


def loading(folder_name: str):
    direc = os.path.join(
        config.DATASET_PATH + folder_name)
    print('total images:{}'.format(len(os.listdir(direc))))
    return os.listdir(direc)


# subfolder = ['rock', 'paper', 'scissors']
# for i in subfolder:
#     x = loading(i)


# %% [markdown]
## To do: Analyzing the datasets (Satyam)
# def analyze():
#     subfolder = ['rock','paper','scissors']
#     for i in subfolder:
#         x = loading(i)
#         for img in x:
#             if (img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")):
#                 img = Image.open("/Users/satyamapantagomeza/DataspellProjects/PaperScissorsRock_byHandmodels/data_original/Combined_dataset/"+i+"/"+img)
#                 plt.imshow(img)
#                 plt.title(i)
#                 plt.axis('Off')
#             plt.show()
# analyze()



# %% [markdown]
# ## To do: Function to transform images to same size (Satyam)
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

# def transform_img():
#     subfolder = ['rock', 'paper', 'scissors']
#     for i in subfolder:
#         x = loading(i)
#         for img in x:
#             img = mpimg.imread("/Users/satyamapantagomeza/DataspellProjects/PaperScissorsRock_byHandmodels/data_original/Combined_dataset/"+i+"/"+img)
#             # np.asarray(img).shape
#             resized_imgs = [T.Resize(size=size)(img) for size in [32,128]]
#             plot(resized_imgs,col_title=["32x32","128x128"])
# transform_img()
# # %% [markdown]
# ## To do: Combining the datasets and saving in a new folder (Satyam)
def analyze():
    subfolder = ['rock','paper','scissors']
    for i in subfolder:
        x = loading(i)
        for img in x:
            if (img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")):
                img = Image.open("/Users/satyamapantagomeza/DataspellProjects/PaperScissorsRock_byHandmodels/data_original/Combined_dataset/"+i+"/"+img)
                plt.imshow(img)
                plt.title(i)
                plt.axis('Off')
            plt.show()
analyze()
# %% [markdown]
# ## To do: Function for Data preprocessing 


# %% [markdown]
# ## To do: Function for Data Augmentation (Shifting, flipping, changing brightness, rotation, adding noise,..) 

# %%

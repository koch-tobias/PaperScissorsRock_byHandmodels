################################################################################################################################
##### Research question: Can transfer learning bring a benefit on the performance of CNN models for Rock, Paper, Scissors? #####
################################################################################################################################

import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm.auto import tqdm
import pandas as pd
from typing import Dict, List, Tuple
from timeit import default_timer as timer 
from pathlib import Path
from loguru import logger
import pickle
from datetime import datetime
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from typing import List, Tuple
from PIL import Image
import random
import time
import math
from data_engineering import split
#########################################################################################
#####                          Function to load the dataset                         #####
#########################################################################################
def load_data(train_dir: str, val_dir: str, weights, num_workers: int, batch_size: int):

    # Get the transforms used to create our pretrained weights
    auto_transforms = weights.transforms()
    logger.info("Get the data transforms that were used to train the model on ImageNet:")
    logger.info(auto_transforms)

    # Create training and valing DataLoaders as well as get a list of class names
    train_dataloader, val_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                                val_dir=val_dir,
                                                                                transform=auto_transforms, # perform same data transforms on our own data as the pretrained model
                                                                                batch_size=batch_size, # set mini-batch size to 32
                                                                                num_workers=num_workers) 


    return train_dataloader, val_dataloader, class_names

#########################################################################################
#####                          Function to create DataLoader                        #####
#########################################################################################
def create_dataloaders(train_dir: str, 
                            val_dir: str, 
                            transform: transforms.Compose, 
                            batch_size: int, 
                            num_workers: int=1
                        ):
    
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)
   
    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(train_data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True
                                )
    val_dataloader = DataLoader(val_data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True
                                )

    return train_dataloader, val_dataloader, class_names

#########################################################################################
#####                      Function to load pretrainend model                       #####
#########################################################################################
def load_pretrained_model(device):

    # Load best available weights from pretraining on ImageNet
    weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
    
    # Load pretrained model with selected weights
    model = torchvision.models.efficientnet_v2_m(weights).to(device)

    return model, weights

#########################################################################################
#####         Function to recreate the classifier layer of the model                #####
#########################################################################################
def recreate_classifier_layer(model: torch.nn.Module, dropout: int, class_names: list, seed: int, device):
    # Freeze all base layers in the "features" section of the model 
    # by setting requires_grad=False
    for param in model.features.parameters():
        param.requires_grad = False

    # Set the manual seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Recreate the classifier layer (one output unit for each class)
    model.classifier = torch.nn.Sequential(
                            torch.nn.Dropout(p=dropout, inplace=True), 
                            torch.nn.Linear(in_features=1280, 
                            out_features=len(class_names), 
                            bias=True)).to(device)
    return model

#########################################################################################
#####              Function to create a folder for the trained model                #####
#########################################################################################
def get_storage_name(targetfolder:str, model_name:str):
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d%m%Y_%H%M")
    folderpath = Path(targetfolder + "/" + model_name + "_model" + "_" + timestampStr)
    folderpath.mkdir(parents=True, exist_ok=True)

    return folderpath

#########################################################################################
#####           Function to save the hyperparameters of the trained model           #####
#########################################################################################
def store_hyperparameters(target_dir_new_model:str,model_name:str, dict:dict):
    folderpath = get_storage_name(target_dir_new_model, model_name)
    
    dict_path = folderpath / ("hyperparameter_dict.pkl")
    with open(dict_path, "wb") as filestore:
        pickle.dump(dict, filestore)
    return folderpath

#########################################################################################
#####                     Function to save the trained model                        #####
#########################################################################################
def store_model(target_dir_new_model: str, model_name: str, hyperparameter_dict: dict, classifier_model:torch.nn.Module, results:dict,batch_size:int, total_train_time:float):
    logger.info("Store model, results and hyperparameters...")
    
    folderpath = store_hyperparameters(target_dir_new_model,model_name, hyperparameter_dict)
    model_path = folderpath / ("model.pkl")
    results_path = folderpath / ("results.pkl")
    summary_path = folderpath / ("summary.pkl")

    # Print a summary using torchinfo (uncomment for actual output)
    model_summary = summary(model=classifier_model, 
                        input_size=(batch_size, 3, 224, 224), # make sure this is "input_size", not "input_shape"
                        col_names=["input_size", "output_size", "num_params", "trainable"],
                        col_width=20,
                        row_settings=["var_names"]
                        )

    with open(summary_path, "wb") as filestore:
        pickle.dump(model_summary, filestore)   

    with open(model_path, "wb") as filestore:
        pickle.dump(classifier_model, filestore)

    with open(results_path, "wb") as filestore:
        pickle.dump(results, filestore)

    df = pd.DataFrame(index=False)
    df["model_name"] = [model_name]
    df["pretrained"] = ["yes"]
    df["epochs"] = [hyperparameter_dict["epochs"]]
    df["seed"] = [hyperparameter_dict["seed"]]
    df["learning_rate"] = [hyperparameter_dict["learning_rate"]]
    df["dropout"] = [hyperparameter_dict["dropout"]]
    df["batch_size"] = [hyperparameter_dict["batch_size"]]
    df["num_workers"] = [hyperparameter_dict["num_workers"]]
    df["total_train_time"] = [total_train_time/60]
    df["train_loss"] = [list(results["train_loss"])[-1]]
    df["train_acc"] = [list(results["train_acc"])[-1]]
    df["val_loss"] = [list(results["val_loss"])[-1]]
    df["val_acc"] = [list(results["val_acc"])[-1]]
    try:
        df_exist = pd.read_csv('models/models_results.csv')
        df_new = pd.concat([df_exist, df], axis=0, ignore_index=True)
        df_new.to_csv('models/models_results.csv')
    except:
        df.to_csv('models/models_results.csv')

    
    logger.info("Model stored!")

    return folderpath

#########################################################################################
#####                          Function to load the model                           #####
#########################################################################################
def get_model(model_folder: str):
    onlyfiles = [f for f in listdir(model_folder) if isfile(join(model_folder, f))]

    model_path = model_folder / onlyfiles[1]
    results_path = model_folder / onlyfiles[2]
    hyperparameters_path = model_folder / onlyfiles[0]

    with open(model_path, "rb") as fid:
        classifier_model = pickle.load(fid)

    with open(results_path, "rb") as fid:
        results = pickle.load(fid)

    with open(hyperparameters_path, "rb") as fid:
        dict = pickle.load(fid)

    logger.info("Model and hyperparameters loaded!")
    logger.info("The model is trained with the following hyperparameters:") 
    logger.info(dict)

    return classifier_model, results, dict

#########################################################################################
#####             Function for plotting the loss curve and the accuracy             #####
#########################################################################################
def plot_loss_curves(results,model_folder,safe_fig=False):
    """Plots training curves of a results dictionary.
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "val_loss": [...],
             "val_acc": [...]}
    """
    loss = results["train_loss"]
    val_loss = results["val_loss"]

    accuracy = results["train_acc"]
    val_accuracy = results["val_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    if safe_fig:
        plt.savefig(model_folder + "/" + "train_loss_acc.png")
    plt.show()

#########################################################################################
#####        Function to make predictions on images from validation set             #####
#########################################################################################
def pred_and_plot_image(model: torch.nn.Module,
                            image_path: str,
                            class_names: List[str] = None,
                            transform=None,
                            ax=None,
                            device="cpu"
                            ):

    # Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255

    # Transform if necessary
    if transform:
        target_image = transform(target_image)

    model.to(device)
    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension 
        target_image_pred = model(target_image.to(device))

    # Convert logits -> prediction probabilities 
    # (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    if ax is None:
        # Plot the image alongside the prediction and prediction probability
        plt.imshow(
            target_image.squeeze().permute(1, 2, 0)
        )  # make sure it's the right size for matplotlib
        if class_names:
            title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
        else:
            title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
        plt.title(title)
        plt.axis(False)
        plt.draw()
        plt.show()
    else:
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.45,
                            top=0.5,
                            wspace=0.4,
                            hspace=0.4)
        # Plot the image alongside the prediction and prediction probability
        ax.imshow(
            target_image.squeeze().permute(1, 2, 0)
        )  # make sure it's the right size for matplotlib
        if class_names:
            title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
        else:
            title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
        ax.set_title(title)
        ax.axis(False)
        plt.draw()

#########################################################################################
#####                 Function to make predictions on single images                 #####
#########################################################################################
def pred_on_single_image(image_path:str, model_folder:str,device):
    class_names = ['paper', 'rock', 'scissors']

    weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
    auto_transforms = weights.transforms()

    trained_model, model_results, dict_hyperparameters = get_model(Path(model_folder))


    pred_and_plot_image(trained_model, image_path, class_names, auto_transforms, device=device)

#########################################################################################
#####                     Function to evaluate an existing model                    #####
#########################################################################################
def eval_existing_model(model_folder:str,validation_folder:str, num_images:int,device):

    trained_model, model_results, dict_hyperparameters = get_model(Path(model_folder))

    plot_loss_curves(model_results,model_folder)

    # Make predictions on random images from validation dataset
    class_names = ['paper', 'rock', 'scissors']

    weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
    auto_transforms = weights.transforms()

    valid_image_path_list = list(Path(validation_folder).glob("*/*.*")) # get list all image paths from val data 
    valid_image_path_sample = random.sample(population=valid_image_path_list, # go through all of the val image paths
                                        k=num_images) # randomly select 'k' image paths to pred and plot

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}
    fig, axes = plt.subplots(math.ceil(num_images/2),2, figsize=(7,7))
    plt.rc('font', **font)

    # Make predictions on and plot the images
    for image_path, ax in zip(valid_image_path_sample,axes.ravel()):
        pred_and_plot_image(model=trained_model, 
                            image_path=image_path,
                            class_names=class_names,
                            transform=auto_transforms,
                            ax = ax,
                            device=device)
    if num_images%2 != 0:
        fig.delaxes(axes[math.ceil(num_images/2)-1,1])
    plt.tight_layout()
    plt.show()

#########################################################################################
#####                    Functions to test model on unseen data                     #####
#########################################################################################
def test_model(model_folder, test_folder):
    trained_model, model_results, dict_hyperparameters = get_model(Path(model_folder))
    image_path_list = list(Path(test_folder).glob("*/*.*"))
    class_names = ['paper', 'rock', 'scissors']
    accuracy = []

    for image_path in image_path_list:
        # Load in image and convert the tensor values to float32
        target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

        # Divide the image pixel values by 255 to get them between [0, 1]
        target_image = target_image / 255

        weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
        auto_transforms = weights.transforms()

        # Transform if necessary
        target_image = auto_transforms(target_image)

        # Turn on model evaluation mode and inference mode
        trained_model.eval()
        with torch.inference_mode():
            # Add an extra dimension to the image
            target_image = target_image.unsqueeze(dim=0)

            # Make a prediction on image with an extra dimension 
            target_image_pred = trained_model(target_image.cuda())

        # Convert logits -> prediction probabilities 
        # (using torch.softmax() for multi-class classification)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

        # Convert prediction probabilities -> prediction labels
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
        pred_class = class_names[target_image_pred_label.item()]
        true_class = image_path.parts[2]  
        if pred_class == true_class:
            accuracy.append(1)
        else:
            accuracy.append(0)
    
    print("Accuracy on test set: " + str(sum(accuracy)/len(accuracy)*100) + " %")

#########################################################################################
#####                Functions to train the transfer learning model                 #####
#########################################################################################
def train_step(model: torch.nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    loss_fn: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer,
                    device
                ) -> Tuple[float, float]:

  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):

      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate  and accumulate loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device
              ) -> Tuple[float, float]:

  # Put model in eval mode
  model.eval() 

  # Setup val loss and val accuracy values
  val_loss, val_acc = 0, 0

  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):
          
          X, y = X.to(device), y.to(device)

          # 1. Forward pass
          val_pred_logits = model(X)

          # 2. Calculate and accumulate loss
          loss = loss_fn(val_pred_logits, y)
          val_loss += loss.item()

          # Calculate and accumulate accuracy
          val_pred_labels = val_pred_logits.argmax(dim=1)
          val_acc += ((val_pred_labels == y).sum().item()/len(val_pred_labels))

  # Adjust metrics to get average loss and accuracy per batch 
  val_loss = val_loss / len(dataloader)
  val_acc = val_acc / len(dataloader)
  return val_loss, val_acc

def train(target_dir_new_model: str,
            model_name: str,
            hyperparameter_dict: dict,
            model: torch.nn.Module, 
            train_dataloader: torch.utils.data.DataLoader, 
            val_dataloader: torch.utils.data.DataLoader, 
            optimizer: torch.optim.Optimizer,
            loss_fn: torch.nn.Module,
            epochs: int,
            batch_size: int,
            device
            ) -> Dict[str, List]:

    # Create empty results dictionary
    results = {"train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": []
            }

    # Start the timer
    start_time = timer()

    # Loop through training and valing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device                                          
                                            )
        val_loss, val_acc = val_step(model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device
            )

        # Print out what's happening
        print(
            f"\nEpoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    # End the timer and print out how long it took
    end_time = timer()

    time.sleep(10)
    total_train_time = end_time-start_time
    print(f"[INFO] Total training time: {total_train_time:.3f} seconds")

    model_folder = store_model(target_dir_new_model, model_name, hyperparameter_dict, model, results,batch_size, total_train_time)

    # Return the filled results at the end of the epochs
    return results, model_folder

def train_new_TransferLearning_model(dataset_path:str, seed:int, learning_rate:float, epochs:int, dropout:float, num_workers:int, batch_size:int):
    train_dir = dataset_path + "/train"
    val_dir = dataset_path + "/val"
    target_dir_new_model = 'models'
    model_name = "TransferLearning"

    dict_hyperparameters = {'seed': seed, 'learning_rate':learning_rate, 'epochs': epochs, 
                        'dropout':dropout, 'num_workers':num_workers,'batch_size': batch_size}

    split(original_dataset_dir='data_original', seed=seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on:")
    logger.info(device)

    # Load pretrained model, weights and the transforms
    model, weights = load_pretrained_model(device)

    # Load data
    train_dataloader, val_dataloader, class_names = load_data(train_dir=train_dir,
                                                                    val_dir=val_dir, 
                                                                    weights=weights, 
                                                                    num_workers=num_workers, 
                                                                    batch_size=batch_size
                                                                    )

    # Recreate classifier layer
    model = recreate_classifier_layer(model=model, 
                                            dropout=dropout, 
                                            class_names=class_names,
                                            seed=seed,
                                            device=device
                                            )

    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Set the random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    # Setup training and save the results
    results, model_folder = train(target_dir_new_model=target_dir_new_model,
                        model_name=model_name,
                        hyperparameter_dict=dict_hyperparameters, 
                        model=model,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=epochs,
                        batch_size=batch_size,
                        device=device
                    )

    plot_loss_curves(results, model_folder,safe_fig=True)
    
    return model_folder

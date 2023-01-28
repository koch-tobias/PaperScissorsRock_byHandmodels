###########################################
##### Research question:              #####
#####Can interpretable model-agnostic explanations (LIME) make our CNN models for Rock, Paper, Scissors more explainable?
###########################################
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, f1_score, recall_score
from PIL import Image
import numpy as np
import os
import torch

import torchvision
from torchvision import transforms
from model_engineering import get_model
from lime import lime_image
from skimage.segmentation import mark_boundaries
from pathlib import Path
from data_engineering import manual_transformation_augmentation


class Lime:
    def __init__(self, model_folder):
        self.model_folder = model_folder

    def get_image(self, path):
        with open(os.path.abspath(path), 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def get_pil_transform(self):
        transf = transforms.Compose([
            transforms.Resize((384, 384))
        ])

        return transf

    def get_preprocess_transform(self):
        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return transf

    def batch_predict(self, images):
        model, model_results, dict_hyperparameters, summary = get_model(self.model_folder)
        model.eval()
        preprocess_transform = self.get_preprocess_transform()

        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        batch = batch.to(device)

        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def single_image_LIME(self, img_path):
        class_names = ['paper', 'rock', 'scissors']
        # Load Image
        img = self.get_image(img_path)
        # Load Model
        model, model_results, dict_hyperparameters, summary = get_model(self.model_folder)

        pill_transf = self.get_pil_transform()

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(image=np.array(pill_transf(img)),
                                                 classifier_fn=self.batch_predict,  # classification function
                                                 top_labels=3,
                                                 hide_color=0,
                                                 num_samples=1000)  # number of images that will be sent to classification function

        # %%
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=3,
                                                    hide_rest=False)
        img_boundry1 = mark_boundaries(temp, mask)
        plt.imshow(img_boundry1)
        plt.show()


def print_model_metrices(model_folder, test_folder):
    trained_model, model_results, dict_hyperparameters, summary = get_model(Path(model_folder))
    image_path_list = list(Path(test_folder).glob("*/*.*"))
    class_names = ['paper', 'rock', 'scissors']
    accuracy = []
    predictions = []
    y_test = []

    for image_path in image_path_list:
        # Load in image and convert the tensor values to float32
        target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

        # Divide the image pixel values by 255 to get them between [0, 1]
        target_image = target_image / 255


        manual_transforms = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        # Transform if necessary
        target_image = manual_transforms(target_image)

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
        predictions.append(target_image_pred_label.item())
        y_test.append(class_names.index(true_class))
        if pred_class == true_class:
            accuracy.append(1)
        else:
            accuracy.append(0)

    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=class_names, cmap='Blues',
                                            colorbar=False)
    plt.savefig(model_folder + '/test_confusion_matrix.png')
    plt.show()

    print("Accuracy on test set: " + str(sum(accuracy) / len(accuracy) * 100) + " %")
    print("Precision on test set " + str(precision_score(y_test, predictions, average='macro')))
    print("Recall on test set " + str(recall_score(y_test, predictions, average='macro')))
    print("F1 Score on test set " + str(f1_score(y_test, predictions, average='macro')))
    # print("Log-Loss on test set " + str(log_loss(y_test, predictions)))

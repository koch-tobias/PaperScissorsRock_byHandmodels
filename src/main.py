from model_engineering import train_new_TransferLearning_model
from model_engineering import plot_loss_acc_curves
from model_engineering import pred_on_example_images
from model_engineering import pred_on_single_image
from model_engineering import test_model
from loguru import logger

if __name__ == "__main__":
    # Define dataset path
    dataset_path = 'data_combined' 
    model_folder = "models/Baseline_model_05122022_1614"
    test_folder = "data_combined/test"
    single_image_path = ""

    # Set parameter for testing
    num_images = 6

    # Set if you want to train a new model or evualate an existing model
    train_new_transferlearning_model = True
    train_new_baseline_model = False
    test_existing_model = False
    prediction_on_single_image = False
    prediction_on_images = False

    if train_new_transferlearning_model:
        logger.info("Start training a new model with Transfer Learning...")
        model_folder = train_new_TransferLearning_model(dataset_path=dataset_path,tf_model=True)
        logger.info("Congratulations, the training was successful!")

    if train_new_baseline_model:
        logger.info("Start training a new model with Transfer Learning...")
        model_folder = train_new_TransferLearning_model(dataset_path=dataset_path,tf_model=False)
        logger.info("Congratulations, the training was successful!")

    if test_existing_model:
        logger.info("Display the train/validation loss/accuracy curves of the trained model:")
        plot_loss_acc_curves(model_folder=model_folder)
        logger.info("Start testing the model..") 
        test_model(model_folder=model_folder, test_folder=test_folder)  
  
    if prediction_on_images:
        pred_on_example_images(model_folder=model_folder, image_folder=test_folder, num_images=num_images)

    if prediction_on_single_image:
        logger.info("Start classifying the given image...") 
        pred_on_single_image(image_path=single_image_path, model_folder=model_folder)


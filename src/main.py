from model_engineering import train_new_model
from model_engineering import plot_loss_acc_curves
from model_engineering import pred_on_example_images
from model_engineering import pred_on_single_image
from model_engineering import test_model
from data_engineering import rgba_to_rgb
from model_evaluation import print_model_metrices, Lime
from loguru import logger

if __name__ == "__main__":

    # Define dataset paths
    dataset_path = 'data_combined' 
    model_folder = "models\TransferLearning_model_24012023_0923"
    test_folder = "data_combined/test"
    single_image_path = 'data_combined/test/scissors/scissors_1.jpg'

    # Set parameter for testing
    num_images = 6 

    # Set if you want to train a new model or which evualation you want to make on an existing model
    train_new_transferlearning_model = False
    train_new_baseline_model = False
    test_existing_model = False
    prediction_on_single_image = False
    prediction_on_images = False
    model_metrices = False
    LIME_single_Image = False
    activate_Augmentation = False

    #Set max one combination=True for training the model with data augmention. 
    #Set activate_Augmentation=False means training without data augmentation
    comb_1=False
    comb_2=False
    comb_3=False
    comb_4=False
    comb_5=False
    comb_6=False
    comb_7=False

    if train_new_transferlearning_model:
        logger.info("Start training a new model with Transfer Learning...")
        model_folder = train_new_model(dataset_path=dataset_path,tf_model=True, activate_augmentation=activate_Augmentation,comb_1=comb_1,comb_2=comb_2,comb_3=comb_3,comb_4=comb_4,comb_5=comb_5,comb_6=comb_6,comb_7=comb_7)
        logger.info("Congratulations, training the Transfer Learning models was successful!")

    if train_new_baseline_model:
        logger.info("Start training a new Baseline model...")
        model_folder = train_new_model(dataset_path=dataset_path,tf_model=False, activate_augmentation=activate_Augmentation,comb_1=comb_1,comb_2=comb_2,comb_3=comb_3,comb_4=comb_4,comb_5=comb_5,comb_6=comb_6,comb_7=comb_7)
        logger.info("Congratulations, training the baseline models was successful!")

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

    if model_metrices:
        logger.info("Start printing some model metrices...")
        print_model_metrices(model_folder=model_folder, test_folder=test_folder)

    if LIME_single_Image:
        logger.info("Start using LIME on given image...")
        LimeInstance = Lime(model_folder=model_folder)
        LimeInstance.single_image_LIME(img_path= single_image_path)



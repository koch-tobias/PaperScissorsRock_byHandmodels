from data_engineering import split
from model_engineering import train_new_TransferLearning_model
from model_engineering import eval_existing_model
from model_engineering import pred_on_single_image
from model_engineering import test_model
from loguru import logger

if __name__ == "__main__":
    # Define dataset path
    dataset_path = 'data_combined'
    model_folder = "models/TransferLearning_model_04122022_1628"
    test_folder = "data_combined/test"
    single_image_path = ""
    original_dataset_dir = 'data_original'

    # Set hyperparameters
    seed = 42
    learning_rate = 0.001
    epochs = 20
    dropout = 0.2
    num_workers = 2
    batch_size = 64

    # Set parameter for evaluation
    num_images_val = 6


    # Set if you want to train a new model or evualate an existing model
    train_new_transferlearning_model = False
    train_new_baseline_model = False
    evaluate_existing_model = True
    prediction_on_single_image = False
    add_new_dataset_to_combined_dataset = False

    if train_new_transferlearning_model:
        logger.info("Start training a new model with Transfer Learning... (This takes about 30 minutes per epoch)")
        model_folder = train_new_TransferLearning_model(dataset_path=dataset_path,
                                                            seed=seed, 
                                                            learning_rate=learning_rate,
                                                            epochs=epochs,
                                                            dropout=dropout,
                                                            num_workers=num_workers,
                                                            batch_size=batch_size)
        logger.info("Congratulations, the training was successful!")
        logger.info("Start evaluating the new model...")  
        test_model(model_folder=model_folder, test_folder=test_folder)  

    if evaluate_existing_model:
        logger.info("Start evaluating the given model..") 
        test_model(model_folder=model_folder, test_folder=test_folder)    
        #eval_existing_model(model_folder=model_folder, validation_folder=test_folder, num_images=num_images_val,device='cpu')

    if prediction_on_single_image:
        logger.info("Start classifying the given image...") 
        pred_on_single_image(image_path=single_image_path, model_folder=model_folder)

    if add_new_dataset_to_combined_dataset:
        split(original_dataset_dir=original_dataset_dir,seed=seed) #DEFAULT SEED=42



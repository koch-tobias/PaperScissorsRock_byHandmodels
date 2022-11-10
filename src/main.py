from model_engineering import train_new_TransferLearning_model
from model_engineering import eval_existing_model
from model_engineering import pred_on_single_image
from loguru import logger

if __name__ == "__main__":
    # Define dataset path
    dataset_path = 'data_combined/dataset_splitted'
    model_folder = "models/model_08112022_2233"
    validation_folder = "data_combined/dataset_splitted/val"
    single_image_path = "data_own_images/paper_Tobi.jpg"


    # Set parameters
    seed = 42
    learning_rate = 0.001
    epochs = 2
    dropout = 0.2
    num_workers = 2
    batch_size = 32
    num_images_val = 6

    # Set if you want to train a new model or evualate an existing model
    train_new_transferlearning_model = True
    train_new_baseline_model = False
    evaluate_existing_model = False
    prediction_on_single_image = True

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
        eval_existing_model(model_folder=model_folder, validation_folder=validation_folder, num_images=num_images_val)  

    if evaluate_existing_model:
        logger.info("Start evaluating the given model..") 
        eval_existing_model(model_folder=model_folder, validation_folder=validation_folder, num_images=num_images_val)

    if prediction_on_single_image:
        logger.info("Start classifying the given image...") 
        pred_on_single_image(image_path=single_image_path, model_folder=model_folder)



from src.mlproject.logger import logging

if __name__ == "__main__":
    logging.info("The execution has started")
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformationConfig
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.components.model_trainer import ModelTrainerConfig
import sys
if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        data_ingestion = DataIngestion()
        #data_ingestion_config = DataIngestionConfig()
        train_data_path,test_data_path =  data_ingestion.initiate_data_ingestion()
        # data_tranformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        ##Model Training 
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr,test_arr)
    except Exception as e :
        logging.info("Custom exception created ")
        raise CustomException(e,sys)
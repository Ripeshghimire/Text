from src.mlproject.logger import logging

if __name__ == "__main__":
    logging.info("The execution has started")
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion

import sys
if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        data_ingestion = DataIngestion()
        #data_ingestion_config = DataIngestionConfig()
        data_ingestion.initiate_data_ingestion()
    except Exception as e :
        logging.info("Custom exception created ")
        raise CustomException(e,sys)
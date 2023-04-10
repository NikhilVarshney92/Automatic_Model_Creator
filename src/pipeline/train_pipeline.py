# pyright: reportMissingModuleSource=false
# pyright: reportShadowedImports=false
# pyright: reportMissingImports=false

import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_cleaning import DataCleaning
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



class TrainPipeline:
    def __init__(self):
        pass

    def initiate_data_built(self):
        try:

            logging.info('Initiate Data Cleaning ...')
            clean_obj = DataCleaning()
            clean_obj.initiateDataCleaning()

            logging.info('Initiate Data Ingestion ...')
            ingest_obj = DataIngestion()
            ingest_obj.saveRawData()

            logging.info('Initiate Data Transforming ...')
            transform_obj = DataTransformation()
            transform_obj.initiateDataTransformation()

        except Exception as e:
            raise CustomException(sys, e)


    def train_pipe(self, model_name):
        try:
            modelObj = ModelTrainer()
            train_results, test_results = modelObj.initiate_model_trainer(model_name)

            return (train_results, test_results)
        
        except Exception as e:
            raise CustomException(sys,e)

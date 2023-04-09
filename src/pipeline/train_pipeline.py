# pyright: reportMissingModuleSource=false
# pyright: reportShadowedImports=false
# pyright: reportMissingImports=false

import sys
from src.exception import CustomException
from src.components.model_trainer import ModelTrainer



class TrainPipeline:
    def __init__(self):
        pass

    def train_pipe(self, model_name):
        try:
            modelObj = ModelTrainer()
            train_results, test_results = modelObj.initiate_model_trainer(model_name)

            return (train_results, test_results)
        
        except Exception as e:
            raise CustomException(sys,e)

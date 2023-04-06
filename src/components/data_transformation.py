# pyright: reportMissingModuleSource=false
# pyright: reportShadowedImports=false
# pyright: reportMissingImports=false

import sys

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.components.model_trainer import ModelTrainer
from src.utils import save_object
from src import constants

class DataTransformation:
    def __init__(self):
        pass

    def getDataTransformer(self, df):
        '''
        This function is responsible for Data transformer.
        
        '''
        try:

            numerical_columns = [feature for feature in df.columns if df[feature].dtype != 'O']
            categorical_columns = [feature for feature in df.columns if df[feature].dtype == 'O']

            logging.info('We have {} numerical features : {}'.format(len(numerical_columns), numerical_columns))
            logging.info('\nWe have {} categorical features : {}'.format(len(categorical_columns), categorical_columns))


            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)

                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiateDataTransformation(self):

        try:
            x_train_df=pd.read_csv(constants.TRAIN_DATA_FILE_PATH)
            x_test_df=pd.read_csv(constants.TEST_DATA_FILE_PATH)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.getDataTransformer(x_train_df)


            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            train_arr=preprocessing_obj.fit_transform(x_train_df)
            test_arr=preprocessing_obj.transform(x_test_df)


            logging.info(f"Saving preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    Obj = DataTransformation()
    train_arr,test_arr = Obj.initiateDataTransformation()
    print(train_arr.shape)
    #modeltrainer=ModelTrainer()
    #print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    
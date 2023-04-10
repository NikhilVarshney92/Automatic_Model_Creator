# pyright: reportMissingModuleSource=false
# pyright: reportShadowedImports=false
# pyright: reportMissingImports=false

import sys
 
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.components.model_trainer import ModelTrainer
from src.components.data_cleaning import DataCleaning
from src.components.data_ingestion import DataIngestion
from src.utils import seperate_cat_num_feature
from src import constants

class DataTransformation:
    def __init__(self):
        pass

    def getDataTransformer(self, numerical_columns, categorical_columns, df, path):
        '''
        This function is responsible for Data transformer.
        
        '''
        try:
            logging.info('Creating Dummies for categorical feature ..')
            # Create Dummies For Categorical Feature
            temp_df1 = pd.get_dummies(df[categorical_columns])
            
            logging.info('Scaling for numberical feature ..')
            # Sacling For Numerical Feature 
            ss = StandardScaler()
            temp_df2 = pd.DataFrame(ss.fit_transform(df[numerical_columns]), columns= numerical_columns)

            logging.info('Merging Transformed Feature ..')
            #Merge ack both tranformed categorical and numberical feature
            transformed_df = pd.concat([temp_df1,temp_df2], axis= 1)
            transformed_df.to_csv(path, index=False, header=True)

            return transformed_df
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiateDataTransformation(self):

        try:
            logging.info('Reading Cleaned train and test Data ..')
            x_train_df=pd.read_csv(constants.xTRAIN_DATA_FILE_PATH)
            x_test_df=pd.read_csv(constants.xTEST_DATA_FILE_PATH)

            logging.info('Seperating numerical and categorical features ..')
            numerical_columns,categorical_columns = seperate_cat_num_feature(x_train_df)

            transformed_x_train = self.getDataTransformer(numerical_columns, categorical_columns, x_train_df, constants.TRANSFORM_xTRAIN_DATA_FILE_PATH)
            transformed_x_test = self.getDataTransformer(numerical_columns, categorical_columns, x_test_df, constants.TRANSFORM_xTEST_DATA_FILE_PATH)

            logging.info('Feature Transformation Completed Successfully !!!')
            return (transformed_x_train, transformed_x_test)


        except Exception as e:
            raise CustomException(e,sys)
    
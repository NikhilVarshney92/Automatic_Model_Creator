# pyright: reportMissingModuleSource=false
# pyright: reportShadowedImports=false

import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from src import constants
from src.components.data_cleaning import DataCleaning


class DataIngestion:

    def __init__(self):
        pass

    def saveRawData(self):
        '''
        This method is to Save Data as Raw, Train, test. 
        '''
        try:
            logging.info('Read the Cleaned dataset as dataframe...')

            df = pd.read_csv(constants.CLEAN_DATA_FILE_PATH)
        
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(constants.TRAIN_DATA_FILE_PATH,index=False,header=True)

            test_set.to_csv(constants.TEST_DATA_FILE_PATH,index=False,header=True)
            logging.info("Ingestion of the data is completed Successfully...")

        
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    Obj = DataCleaning()
    Obj.initiateDataCleaning()
    Ovj2 = DataIngestion()
    Ovj2.saveRawData()
    pass
    

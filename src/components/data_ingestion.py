# pyright: reportMissingModuleSource=false
# pyright: reportShadowedImports=false

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataPathConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:

    def __init__(self):

        self.path_config = DataPathConfig()

    def saveRawData(self):
        '''
        This method is to Save Data as Raw, Train, test. 
        '''
        try:
            logging.info('Read the dataset as dataframe...')
            df = pd.read_csv('Data\StudentsPerformance.csv')

            os.makedirs(os.path.dirname(self.path_config.train_data_path),exist_ok=True)

            df.to_csv(self.path_config.raw_data_path,index=False,header=True)

            logging.info('Raw Data Saved Successfully...')
        
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.path_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.path_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed Successfully...")

            return(
                self.path_config.train_data_path,
                self.path_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    pass
    

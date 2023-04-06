# pyright: reportMissingModuleSource=false
# pyright: reportShadowedImports=false
# pyright: reportMissingImports=false

import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src import constants


class DataCleaning:
    def __init__(self):
        pass

    def cleanNulls(self, df):

        logging.info('Removing Nulls')
        return df.dropna(inplace = True)
    
    
    def cleanDuplicates(self, df):

        logging.info('Removing Duplicates')
        return df.drop_duplicates(inplace = True)
    
    
    def cleanirrevelentCols(self, df, columns: list):

        logging.info('Removing Useless Columns')
        return df.drop(columns, axis=1, inplace=True)
    

    def initiateDataCleaning(self):
        '''
        This method is used to perform cleaning Operation over data.
        '''
        try:
            
            df = pd.read_csv(constants.RAW_DATA_FILE_PATH)

            logging.info('Checking for Nulls')
            if df.isnull().sum().any() == True:
                df = self.cleanNulls(df)
            else:
                pass

            logging.info('Checking for Duplicates')
            if df.duplicated().any() == True:
                df = self.cleanDuplicates(df)
            else:
                pass

            logging.info('Checking for Irrevelent Columns')
            unique_list = list(df.nunique())
            single_value_col = [ df.columns[i] for i in range(len(unique_list)) if unique_list[i] == 1]
            if single_value_col:
                df = self.cleanirrevelentCols(df)
            else:
                pass
            
            logging.info('Converting Columns to their best dtypes formats')
            df = df.convert_dtypes()

        except Exception as e:
            raise CustomException(e,sys)


        df.to_csv(constants.CLEAN_DATA_FILE_PATH,index=False,header=True)

if __name__=="__main__":
    #Obj = DataCleaning()
    #df = Obj.initiateDataCleaning()
    #print(df.head())
    pass
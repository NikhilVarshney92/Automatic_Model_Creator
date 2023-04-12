# pyright: reportMissingModuleSource=false
# pyright: reportShadowedImports=false
# pyright: reportMissingImports=false

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from src.exception import CustomException
from src.logger import logging
from src import constants
from src.utils import make_folder

class DataVisualization:
    def __init__(self):
        pass

    def initiateVisualization(self, model_name):

        try:
            logging.info('Fetching training result dataframe ..')
            PATH = make_folder(constants.RESULT_DATA_FOLDER_PATH, model_name)
            train_result = pd.read_csv(PATH+'/train_result_df.csv')

            logging.info('Creating satterplot .. ')
            sns_plot = sns.scatterplot(data=train_result, x= 'ACTUAL', y ='PRED')
            fig = sns_plot.get_figure()

            IMG_PATH = make_folder(constants.RESULT_IMG_FOLDER_PATH, model_name)
            img_path = IMG_PATH+"/train_result.png"
            logging.info('Image is {}'.format(img_path))
            fig.savefig(img_path)

            return img_path

        except Exception as e:
            raise CustomException(sys, e)
        
    def readImage(self, model_name):
        try:
            pass
        except Exception as e:
            raise CustomException(sys, e)

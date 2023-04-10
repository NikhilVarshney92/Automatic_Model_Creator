# pyright: reportMissingModuleSource=false
# pyright: reportShadowedImports=false
# pyright: reportMissingImports=false

import sys

import statsmodels.api as sm
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src import constants

from src.utils import evaluate_models

class ModelTrainer:
    def __init__(self):
        pass

    def linear_regression_model(self, X_train, y_train, X_test, y_test):

        # Use OLS model for finding feature based on p-value
        try:
            logging.info('Creating Linear Regression Model ..')
            X_new = sm.add_constant(X_train)
            ols_model = sm.OLS(y_train,X_new)
            temp_results = ols_model.fit()
            p_values_columns = list(temp_results.pvalues[temp_results.pvalues <= constants.P_VALUE].index)
            
            if 'const' in p_values_columns:
                p_values_columns.remove('const')
            
            model = LinearRegression()

            logging.info('Selecting Features Based on p_value ..')
            X_train = X_train[p_values_columns]
            X_test = X_test[p_values_columns]

            train_model_score_dict, test_model_score_dict = evaluate_models(X_train, y_train, X_test, y_test, model, {}, 'regression')

            logging.info('Linear Regression Model runs successfully !!!')
            
            return (train_model_score_dict, test_model_score_dict)

        except Exception as e:
            raise CustomException(e, sys)

    def random_forest_regression_model(self, X_train, y_train, X_test, y_test):

        try:
            params_rf = {
                    #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                }
            logging.info('Creating Random Forest Regression Model ..')
            
            model = RandomForestRegressor()

            logging.info('Evaluating model ..')
            train_model_score_dict, test_model_score_dict = evaluate_models(X_train, y_train, X_test, y_test, model, params_rf, 'regression')

            logging.info('Random Forest Regression Model runs successfully !!!')
            
            return (train_model_score_dict, test_model_score_dict)

        except Exception as e:
            raise CustomException(e, sys)
        
    def decision_tree_regression_model(self, X_train, y_train, X_test, y_test):
        
        try:
            params_rf = {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                }
            logging.info('Creating Decision tree Regression Model ..')
            
            model = DecisionTreeRegressor()

            logging.info('Evaluating model ..')
            train_model_score_dict, test_model_score_dict = evaluate_models(X_train, y_train, X_test, y_test, model, params_rf, 'regression')

            logging.info('Decision Tree Regression Model runs successfully !!!')
            
            return (train_model_score_dict, test_model_score_dict)

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def gradientboost_regression_model(self, X_train, y_train, X_test, y_test):

        try:
            params_rf = {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            
            logging.info('Creating GradientBoosting Regression Model ..')
            
            model = GradientBoostingRegressor()

            logging.info('Evaluating model ..')
            train_model_score_dict, test_model_score_dict = evaluate_models(X_train, y_train, X_test, y_test, model, params_rf, 'regression')

            logging.info('GradientBoosting Regression Model runs successfully !!!')
            
            return (train_model_score_dict, test_model_score_dict)

        except Exception as e:
            raise CustomException(e, sys)
        

    def XGB_regression_model(self, X_train, y_train, X_test, y_test):

        try:
            params_rf = {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [4,8,16,32,64,128,256]
                }
            logging.info('Creating XGB Regression Model ..')
            
            model = XGBRegressor()

            logging.info('Evaluating model ..')
            train_model_score_dict, test_model_score_dict = evaluate_models(X_train, y_train, X_test, y_test, model, params_rf, 'regression')

            logging.info('XGBoosting Regression Model runs successfully !!!')
            
            return (train_model_score_dict, test_model_score_dict)

        except Exception as e:
            raise CustomException(e, sys)
        

    def catboost_regression_model(self, X_train, y_train, X_test, y_test):

        try:
            params_rf = {
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                }
            logging.info('Creating CATBoost Regression Model ..')
            
            model = CatBoostRegressor()

            logging.info('Evaluating model ..')
            train_model_score_dict, test_model_score_dict = evaluate_models(X_train, y_train, X_test, y_test, model, params_rf, 'regression')

            logging.info('CATBoosting Regression Model runs successfully !!!')
            
            return (train_model_score_dict, test_model_score_dict)

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def AdaBoost_regression_model(self, X_train, y_train, X_test, y_test):

        try:
            params_rf = {
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
            }
            logging.info('Creating ADABoost Regression Model ..')
            
            model = AdaBoostRegressor()

            logging.info('Evaluating model ..')
            train_model_score_dict, test_model_score_dict = evaluate_models(X_train, y_train, X_test, y_test, model, params_rf, 'regression')

            logging.info('AdaBoost Regression Model runs successfully !!!')
            
            return (train_model_score_dict, test_model_score_dict)

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_model_trainer(self, model_name):
        try:

            X_train = pd.read_csv(constants.TRANSFORM_xTRAIN_DATA_FILE_PATH)
            y_train = pd.read_csv(constants.yTRAIN_DATA_FILE_PATH)
            X_test = pd.read_csv(constants.TRANSFORM_xTEST_DATA_FILE_PATH)
            y_test = pd.read_csv(constants.yTEST_DATA_FILE_PATH)

            if model_name == 'LinearRegression':
                train_model_score, test_model_score = self.linear_regression_model(X_train, y_train, X_test, y_test)
            elif model_name == 'RandomForestRegression':
                train_model_score, test_model_score = self.random_forest_regression_model(X_train, y_train, X_test, y_test)
            elif model_name == 'DecisionTreeRegression':
                train_model_score, test_model_score = self.decision_tree_regression_model(X_train, y_train, X_test, y_test)
            elif model_name == 'GradientBoostRegression':
                train_model_score, test_model_score = self.gradientboost_regression_model(X_train, y_train, X_test, y_test)
            elif model_name == 'AdaBoostRegression':
                train_model_score, test_model_score = self.AdaBoost_regression_model(X_train, y_train, X_test, y_test)
            elif model_name == 'XGBoostRegression':
                train_model_score, test_model_score = self.XGB_regression_model(X_train, y_train, X_test, y_test)
            elif model_name == 'CatBoostRegression':
                train_model_score, test_model_score = self.catboost_regression_model(X_train, y_train, X_test, y_test)
            

            return (train_model_score, test_model_score)
            
        except Exception as e:
            raise CustomException(e,sys)
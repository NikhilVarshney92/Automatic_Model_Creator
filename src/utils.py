# pyright: reportMissingModuleSource=false
# pyright: reportShadowedImports=false
# pyright: reportMissingImports=false

import os
import sys
import pickle
import numpy as np
# regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_pinball_loss, d2_pinball_score,d2_absolute_error_score 
# classification metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, log_loss, precision_score, confusion_matrix, classification_report, jaccard_score, roc_auc_score

from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, model, param, model_type):
    
    try:
        logging.info('Finding best params for our {} model .. '.format(model))
        gs = GridSearchCV(model, param, cv=3)
        gs.fit(X_train,np.ravel(y_train))
        model.set_params(**gs.best_params_)

        logging.info('Fitting best params in our {} model .. '.format(model))
        model.fit(X_train,np.ravel(y_train))

        logging.info('Predicting on x_train and x_test dataset .. ')
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if model_type == 'regression':
            train_model_score = metrices_regression(y_train, y_train_pred)
            test_model_score = metrices_regression(y_test, y_test_pred)

        elif model_type == 'classification':
            train_model_score = metrices_classification(y_train, y_train_pred)
            test_model_score = metrices_classification(y_test, y_test_pred)
        

        return (train_model_score, test_model_score)

    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def metrices_regression(test, pred):
    results = dict()
    try:
        logging.info('Creating regression metrics dictionaray .. ')
        results['mean_absolute_error'] = mean_absolute_error(test, pred)
        results['mean_squared_error'] = mean_squared_error(test, pred)
        results['mean_pinball_loss'] = mean_pinball_loss(test, pred)
        results['r2_score'] = r2_score(test, pred)
        results['explained_variance_score'] = explained_variance_score(test, pred)
        results['d2_absolute_error_score'] = d2_absolute_error_score(test, pred)
        results['d2_pinball_score'] = d2_pinball_score(test, pred)

        return results

    except Exception as e:
        raise CustomException(e, sys)
    
def metrices_classification(test, pred):
    results = dict()
    try:
        logging.info('Creating Classification metrics dictionaray .. ')
        results['accuracy_score'] = accuracy_score(test, pred)
        results['f1_score'] = f1_score(test, pred)
        results['recall_score'] = recall_score(test, pred)
        results['log_loss'] = log_loss(test, pred)
        results['precision_score'] = precision_score(test, pred)
        results['confusion_matrix'] = confusion_matrix(test, pred)
        results['classification_report'] = classification_report(test, pred)
        results['jaccard_score'] = jaccard_score(test, pred)
        results['roc_auc_score'] = roc_auc_score(test, pred)

        return results

    except Exception as e:
        raise CustomException(e, sys)
    

def seperate_cat_num_feature(df):
    try:
        numerical_columns = [feature for feature in df.columns if df[feature].dtype != 'O']
        categorical_columns = [feature for feature in df.columns if df[feature].dtype == 'O']

        logging.info('We have {} numerical features : {}'.format(len(numerical_columns), numerical_columns))
        logging.info('\nWe have {} categorical features : {}'.format(len(categorical_columns), categorical_columns))



        return (numerical_columns,categorical_columns)
    
    except Exception as e:
        raise CustomException(e, sys)
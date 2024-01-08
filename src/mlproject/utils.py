import pickle
import numpy 
import pandas
import os 
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb" )as file:
            pickle.dump(obj,file)
    except Exception as e:
        raise CustomException(e,sys)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            params = param.get(model_name, {})  # Get the hyperparameters for the current model

            if params:  # Check if there are hyperparameters to tune
                gs = GridSearchCV(model, params, cv=3)
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = f1_score(y_train, y_train_pred, average='weighted')
            test_model_score = f1_score(y_test, y_test_pred, average='weighted')

            report[model_name] = {
                'train_score': train_model_score,
                'test_score': test_model_score,
                'best_params': getattr(model, 'best_params_', None)
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
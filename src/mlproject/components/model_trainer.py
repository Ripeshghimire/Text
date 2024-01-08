from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from sklearn.metrics import accuracy_score,precision_score,f1_score,roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
import os 
import sys 
from dataclasses import dataclass
from src.mlproject.utils import save_object,evaluate_models
@dataclass
class ModelTrainerConfig:
    trained_model_file = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model.trainer.config = ModelTrainerConfig()
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split the data into training and test input data")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "KNeighborsClassifier": KNeighborsClassifier(),
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "SVC": SVC(),
            }
            params_classifiers = {
                    "KNeighborsClassifier": {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                    },
                    "DecisionTreeClassifier": {
                        'criterion': ['gini', 'entropy'],
                        'splitter': ['best', 'random'],
                        'max_features': ['sqrt', 'log2', None]
                    },
                    "RandomForestClassifier": {
                        'n_estimators': [8, 16, 32, 64, 128, 256],
                        'criterion': ['gini', 'entropy'],
                        'max_features': ['sqrt', 'log2', None]
                    },
                    "AdaBoostClassifier": {
                        'n_estimators': [8, 16, 32, 64, 128, 256],
                        'learning_rate': [.1, .01, 0.5, .001],
                        'algorithm': ['SAMME', 'SAMME.R']
                    },
                    "SVC": {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'degree': [2, 3, 4],
                        'gamma': ['scale', 'auto']
                    }
                }
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params_classifiers)
            ##getting the best model 
            best_model_score = max(sorted(model_report.values()))

             ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print("This is the best model:")
            print(best_model_name)

            model_names = list(params_classifiers.keys())

            actual_model=""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_params = params_classifiers[actual_model]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            f1_score = f1_score(y_test, predicted)
            return f1_score

        except Exception as e:
            raise CustomException(e,sys)
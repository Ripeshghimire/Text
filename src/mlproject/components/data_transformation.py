import sys
from dataclasses import dataclass ##dataclass module in python provides a decorator and functions automatically a
                                #dding a special methods such as __init__() and __repr__()
import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import os
from src.mlproject.utils import save_object
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re 
lm = WordNetLemmatizer()

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.datatransformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        '''This function is responsible for data transformation'''
        try:
            pass
            column = 'title'
            cat_Pipeline = Pipeline(steps =[
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ("encoder",OneHotEncoder(handle_unknown='ignore'))
            ])
            logging.info(f'columns:{column}')
            return cat_Pipeline
        except Exception as e :
            raise CustomException(e,sys)
        
    ###Applying preprocessing in the data 
    def preprocess_text(self,text):
        review = re.sub('[^A-za-z]',' ',str(text))
        review = review.lower()
        review = review.split()
        review = [lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        return pd.Series(review)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading the train and test file")
            processing = self.get_data_transformer_object()
            target_Column = "label"
            dropping_columns = ["id","author","text",'label']
            title_column = "title"
            train_df['title'] = train_df["title"].apply(self.preprocess_text)
            test_df['title'] = test_df["title"].apply(self.preprocess_text)
            #dividing the train datset to independent and dependent features
            input_features_train_df = train_df.drop (dropping_columns, axis=1)
            target_feature_train_df  = train_df[target_Column]
            #dividing the test datset to independent and dependent features
            input_features_test_df = test_df.drop(dropping_columns, axis=1)
            target_feature_test_df  = test_df[target_Column]
            logging.info("Applying Preprocessing on training and test dataframe")


            input_feature_train_arr = processing.fit_transform(input_features_train_df)
            input_feature_test_arr = processing.transform(input_features_test_df) #using transform in the test data to handle data leakage
             

             #combining the data for traiin
            print(input_feature_train_arr.shape)
            print(target_feature_train_df.shape) 
            print(input_feature_test_arr) 
            print(target_feature_test_df) 
            train_arr = np.c_[
            input_feature_train_arr, np.array(target_feature_train_df).reshape(-1, 14560)
            ]

            #combining the data for test
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df).reshape(-1,6240)
            ]
            logging.info(f"Saved Preprocessing object")
            save_object( ### dumping the pickle file using the save object function defined in utils.py
                file_path=self.datatransformation_config.preprocessor_obj_file,
                obj = processing
            )
            #returing the data
            return(
                train_arr,
                test_arr,
                self.datatransformation_config.preprocessor_obj_file
            )
        except Exception as e:
            raise CustomException(e,sys)
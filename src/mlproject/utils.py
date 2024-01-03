import pickle
import numpy 
import pandas
import os 
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb" )as file:
            pickle.dump(obj,file)
    except Exception as e:
        raise CustomException(e,sys)
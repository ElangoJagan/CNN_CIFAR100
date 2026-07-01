from src.exception import CustomException
from src.logger import Logger
import sys
import os 
import numpy as np

from sklearn.model_selection import train_test_split

_logger_obj = Logger('Preprocessor')
logger = _logger_obj.get_logger()
class Preprocessor:
    
    
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        
    
    def load_data(self):
        try:
            if not os.path.exists(self.raw_data_path):
                raise FileNotFoundError(f'Raw data file not found at {self.raw_data_path}')
            
            data = np.load(self.raw_data_path)
            
            x_train = data['x_train']
            y_train = data['y_train']
            x_test = data['x_test']
            y_test = data['y_test']
            
            logger.info(f"Raw data loaded: x_train={x_train.shape}, x_test={x_test.shape}")
            
            return x_train, y_train, x_test, y_test
        except Exception as e:
            raise CustomException(e, sys)       
    
    @staticmethod
    def normalize_data(x_train, x_test):
        try:
            x_train = x_train.astype(np.float32)/255
            x_test = x_test.astype(np.float32)/255
            logger.info("Data normalized to range 0-1")
            return x_train, x_test  
        except Exception as e:
            raise CustomException(e, sys)                               
    
    @staticmethod
    def split_train_val(x, y, val_size, random_state):
        try:
            x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = val_size, random_state=random_state, stratify = y)
            logger.info(f"Train/Val split done: train={x_train.shape[0]}, val={x_val.shape[0]}")
            return x_train, x_val, y_train, y_val
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_preprocessing(self, val_size = 0.1, random_state = 42):
        
        try:
            
            logger.info('Preprocessing initiated')
            x_train, y_train, x_test, y_test = self.load_data()
            
            x_train, x_test = self.normalize_data(x_train, x_test)
            
            x_train, x_val, y_train, y_val = self.split_train_val(x_train, y_train, val_size, random_state)
            
            logger.info('Preprocessing completed successfully')
            return x_train, x_val, x_test, y_train, y_val, y_test
        except Exception as e:
            raise CustomException(e, sys)
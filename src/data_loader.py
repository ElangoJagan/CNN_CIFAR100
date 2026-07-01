from src.exception import CustomException
from src.logger import Logger
from src.configfile import DataLoaderConfig

import sys
import os
import numpy as np

from tensorflow.keras.datasets import cifar100

_logger_obj = Logger('DataLoader')
logger = _logger_obj.get_logger()


class DataLoader:

    def __init__(self):
        logger.info('DataLoader Initialized')
        self.config = DataLoaderConfig()

    def load_data(self):
        try:
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()

            # fix label shape: (N,1) -> (N,)
            y_train = y_train.flatten()
            y_test = y_test.flatten()

            logger.info(f"Data loaded: x_train={x_train.shape}, "
                        f"y_train={y_train.shape}, x_test={x_test.shape}, "
                        f"y_test={y_test.shape}")

            return x_train, y_train, x_test, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def save_data(self, x_train, y_train, x_test, y_test):
        try:
            path = self.config.raw_data_path
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, x_train=x_train, y_train=y_train,x_test=x_test, y_test=y_test)
            logger.info(f"Data saved at: {path}")
            return path
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_loader(self):
        try:
            
            path = self.config.raw_data_path

            # if already saved, skip load + save entirely
            if os.path.exists(path):
                logger.info(f"Raw data already exists at {path} -- skipping download/save")
                return path
            
            x_train, y_train, x_test, y_test = self.load_data()
            path = self.save_data(x_train, y_train, x_test, y_test)
            return path
        except Exception as e:
            raise CustomException(e, sys)
from src.exception import CustomException
from src.logger import Logger
import sys
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

_logger_obj = Logger('CNNModel')
logger = _logger_obj.get_logger()

class CNNModel:
    CONV1_FILTERS = 32
    CONV2_FILTERS= 64
    CONV3_FILTERS = 128
    KERNEL_SIZE = (3,3)
    POOL_SIZE = (2,2)
    DENSE_UNITS = 256
    DROPOUT_RATE = 0.5
    NUM_CLASSES = 100
    INPUT_SHAPE = (32,32,3)
    LEARNING_RATE = 0.001

    def __init__(self):
        logger.info('CNN Model Initialized')
        self.model = None
    
    def build_model(self):
        try:
            model = Sequential()
            
            #1. Conv Block 1
            model.add(Conv2D(self.CONV1_FILTERS, self.KERNEL_SIZE, activation = 'relu',padding = 'same',input_shape = self.INPUT_SHAPE))
            model.add(MaxPooling2D(self.POOL_SIZE))
            
            #2. Conv Block 2
            model.add(Conv2D(self.CONV2_FILTERS, self.KERNEL_SIZE, activation = 'relu',padding = 'same'))
            model.add(MaxPooling2D(self.POOL_SIZE))
            
            #3. Conv Block 3
            model.add(Conv2D(self.CONV3_FILTERS, self.KERNEL_SIZE, activation = 'relu',padding = 'same'))
            model.add(MaxPooling2D(self.POOL_SIZE))
            
            
            #flatten and Dense 
            model.add(Flatten())
            model.add(Dense(self.DENSE_UNITS, activation = 'relu'))
            model.add(Dropout(self.DROPOUT_RATE))
            
            #Outpput layer:
            model.add(Dense(self.NUM_CLASSES, activation = 'softmax'))
            
            self.model = model
            logger.info("Model architecture built successfully")
            logger.info(f"Total layers: {len(model.layers)}")

            return self.model
        except Exception as e:
            raise CustomException(e, sys)
        
    def compile_model(self):
        try:
            if self.model is None:
                raise ValueError("Model architecture is not built yet. Call build_model() first."  )
            
            self.model.compile(
                optimizer = Adam(learning_rate = self.LEARNING_RATE),
                loss= 'sparse_categorical_crossentropy',
                metrics = ['accuracy']
                
            )
            logger.info("Model compiled successfully")
            return self.model
        except Exception as e:
            raise CustomException(e, sys)   
        
        
    def get_model(self):
        try:
            self.build_model()
            self.compile_model()
            return self.model
        except Exception as e:  
            raise CustomException(e, sys)

if __name__ == '__main__':
    cnn = CNNModel()
    model = cnn.get_model()
    model.summary()
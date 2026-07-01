from src.exception import CustomException
from src.logger import Logger
import sys

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

_logger_obj = Logger('Trainer')
logger = _logger_obj.get_logger()


class Trainer:
    EPOCHS = 30
    BATCH_SIZE = 64
    PATIENCE = 5
    CHECKPOINT_PATH = 'artifacts/models/best_model.keras'
    
    def __init__(self,model):
        logger.info('Trainer Initialized')
        self.model = model
        
    def get_callbacks(self):
        try:
            early_stop =  EarlyStopping(
                monitor = 'val_loss',
                patience = self.PATIENCE,
                restore_best_weights = True
            )
            
            checkpoint = ModelCheckpoint(
                filepath = self.CHECKPOINT_PATH,
                monitor = 'val_loss',
                save_best_only= True,
                verbose = 1
            )
            logger.info("Callbacks created: EarlyStopping and ModelCheckpoint")
            return [early_stop, checkpoint]
        except Exception as e:
            raise CustomException(e,sys)
    
    def train(self, x_train, y_train, x_val, y_val):
        try:
            callbacks = self.get_callbacks()
            logger.info(f"Training started -- epochs={self.EPOCHS}, batch_size={self.BATCH_SIZE}")
            
            history = self.model.fit(
                x_train, y_train,
                validation_data = (x_val, y_val),
                epochs = self.EPOCHS,
                batch_size = self.BATCH_SIZE,
                callbacks = callbacks,
                verbose = 1
            )
            
            logger.info("Training completed successfully")
            return history
        except Exception as e:
            raise CustomException(e,sys)

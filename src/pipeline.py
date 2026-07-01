from src.exception import CustomException
from src.logger import Logger
from src.configfile import DataLoaderConfig
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
import sys

_logger_obj = Logger('Pipeline')
logger = _logger_obj.get_logger()

class Pipeline:
    
    def __init__(self):
        self.config = DataLoaderConfig()
        
    def run_pipeline(self):
        try:
            logger.info("=== Data pipeline started ===")
            
            loader = DataLoader()
            path = loader.initiate_data_loader()
            logger.info(f"Step 1 done -- raw data saved at: {self.config.raw_data_path}")
            
            preprocessor = Preprocessor(raw_data_path = path)
            x_train, x_val, x_test, y_train, y_val, y_test = preprocessor.initiate_preprocessing()
            logger.info("Step 2 done -- data normalized and split")

            logger.info("=== Data pipeline finished successfully ===")

            return x_train, x_val, x_test, y_train, y_val, y_test
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    pipeline = Pipeline()
    x_train, x_val, x_test, y_train, y_val, y_test = pipeline.run_pipeline()

    print(f"x_train: {x_train.shape}")
    print(f"x_val:   {x_val.shape}")
    print(f"x_test:  {x_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_val:   {y_val.shape}")
    print(f"y_test:  {y_test.shape}")
    
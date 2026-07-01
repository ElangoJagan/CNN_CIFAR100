from src.exception import CustomException
from src.logger import Logger
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model import CNNModel
from src.trainer import Trainer
from src.configfile import DataLoaderConfig

import sys

_logger_obj = Logger('Pipeline')
logger = _logger_obj.get_logger()


class Pipeline:

    def __init__(self):
        self.config = DataLoaderConfig()

    def run_data_pipeline(self):
        try:
            logger.info("=== Data pipeline started ===")

            # Step 1: Load raw data
            loader = DataLoader()
            raw_data_path = loader.initiate_data_loader()
            logger.info(f"Step 1 done -- raw data saved at: {raw_data_path}")

            # Step 2: Preprocess -- normalize, split
            preprocessor = Preprocessor(raw_data_path=raw_data_path)
            x_train, x_val, x_test, y_train, y_val, y_test = preprocessor.initiate_preprocessing()
            logger.info("Step 2 done -- data normalized and split")

            logger.info("=== Data pipeline finished successfully ===")

            return x_train, x_val, x_test, y_train, y_val, y_test

        except Exception as e:
            raise CustomException(e, sys)

    def run_training_pipeline(self, x_train, y_train, x_val, y_val):
        try:
            logger.info("=== Training pipeline started ===")

            # Step 3: Build + compile model
            cnn = CNNModel()
            model = cnn.get_model()
            logger.info("Step 3 done -- model built and compiled")

            # Step 4: Train model
            trainer = Trainer(model)
            history = trainer.train(x_train, y_train, x_val, y_val)
            logger.info("Step 4 done -- training completed")

            logger.info("=== Training pipeline finished successfully ===")

            return model, history

        except Exception as e:
            raise CustomException(e, sys)

    def run_full_pipeline(self):
        try:
            x_train, x_val, x_test, y_train, y_val, y_test = self.run_data_pipeline()
            model, history = self.run_training_pipeline(x_train, y_train, x_val, y_val)
            return model, history, x_test, y_test
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = Pipeline()
    model, history, x_test, y_test = pipeline.run_full_pipeline()

    print("\nTraining finished.")
    print(f"Final train accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final val accuracy:   {history.history['val_accuracy'][-1]:.4f}")
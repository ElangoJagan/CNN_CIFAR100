from src.exception import CustomException
from src.logger import Logger
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model import CNNModel
from src.trainer import Trainer
from src.evaluator import Evaluator
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

            loader = DataLoader()
            raw_data_path = loader.initiate_data_loader()
            logger.info(f"Step 1 done -- raw data saved at: {raw_data_path}")

            preprocessor = Preprocessor(raw_data_path=raw_data_path)
            x_train, x_val, x_test, y_train, y_val, y_test = preprocessor.initiate_preprocessing()
            logger.info("Step 2 done -- data normalized and split")

            logger.info("=== Data pipeline finished successfully ===")
            return x_train, x_val, x_test, y_train, y_val, y_test

        except Exception as e:
            raise CustomException(e, sys)

    def run_training_pipeline(self, x_train, y_train, x_val, y_val, use_augmentation=False):
        try:
            logger.info("=== Training pipeline started ===")

            cnn = CNNModel()
            model = cnn.get_model()
            logger.info("Step 3 done -- model built and compiled")

            trainer = Trainer(model)
            history = trainer.train(x_train, y_train, x_val, y_val, use_augmentation=use_augmentation)
            logger.info("Step 4 done -- training completed")

            logger.info("=== Training pipeline finished successfully ===")
            return model, history

        except Exception as e:
            raise CustomException(e, sys)

    def run_evaluation_pipeline(self, model, x_test, y_test, history):
        try:
            logger.info("=== Evaluation pipeline started ===")

            evaluator = Evaluator(model, x_test, y_test)
            results = evaluator.initiate_evaluation(history)

            logger.info(f"Step 5 done -- {evaluator}")
            logger.info("=== Evaluation pipeline finished successfully ===")
            return results

        except Exception as e:
            raise CustomException(e, sys)

    def run_full_pipeline(self, use_augmentation=False):
        try:
            x_train, x_val, x_test, y_train, y_val, y_test = self.run_data_pipeline()
            model, history = self.run_training_pipeline(
                x_train, y_train, x_val, y_val, use_augmentation=use_augmentation
            )
            results = self.run_evaluation_pipeline(model, x_test, y_test, history)
            return model, history, results
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = Pipeline()

    # Set True to test augmentation, False for plain training
    model, history, results = pipeline.run_full_pipeline(use_augmentation=True)

    print("\n" + "="*50)
    print(f"TEST ACCURACY: {results['test_accuracy']:.4f}")
    print("="*50)
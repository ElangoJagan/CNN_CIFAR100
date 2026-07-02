from src.exception import CustomException
from src.logger import Logger
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model import CNNModel
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.experiment_tracker import ExperimentTracker
from src.misclassification_analyzer import MisclassificationAnalyzer
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

            preprocessor = Preprocessor(raw_data_path=raw_data_path)
            x_train, x_val, x_test, y_train, y_val, y_test = preprocessor.initiate_preprocessing()

            logger.info("=== Data pipeline finished ===")
            return x_train, x_val, x_test, y_train, y_val, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def run_training_pipeline(self, x_train, y_train, x_val, y_val, use_augmentation=False):
        try:
            logger.info("=== Training pipeline started ===")
            cnn = CNNModel()
            model = cnn.get_model()

            trainer = Trainer(model)
            history = trainer.train(x_train, y_train, x_val, y_val, use_augmentation=use_augmentation)

            logger.info("=== Training pipeline finished ===")
            return model, history
        except Exception as e:
            raise CustomException(e, sys)

    def run_evaluation_pipeline(self, model, x_test, y_test, history):
        try:
            logger.info("=== Evaluation pipeline started ===")
            evaluator = Evaluator(model, x_test, y_test)
            results = evaluator.initiate_evaluation(history)
            logger.info("=== Evaluation pipeline finished ===")
            return evaluator, results
        except Exception as e:
            raise CustomException(e, sys)

    def run_tracking_pipeline(self, experiment_name, use_augmentation, history, results):
        try:
            logger.info("=== Tracking pipeline started ===")
            tracker = ExperimentTracker()
            record = tracker.log_experiment(
                experiment_name=experiment_name,
                epochs=Trainer.EPOCHS,
                batch_size=Trainer.BATCH_SIZE,
                augmentation=use_augmentation,
                train_accuracy=history.history['accuracy'][-1],
                val_accuracy=history.history['val_accuracy'][-1],
                test_accuracy=results['test_accuracy']
            )
            logger.info("=== Tracking pipeline finished ===")
            return record
        except Exception as e:
            raise CustomException(e, sys)

    def run_misclassification_pipeline(self, evaluator):
        try:
            logger.info("=== Misclassification analysis started ===")
            analyzer = MisclassificationAnalyzer(evaluator.y_test, evaluator.y_pred)
            top_pairs = analyzer.initiate_analysis()

            logger.info("=== Misclassification analysis finished ===")
            return top_pairs
        except Exception as e:
            raise CustomException(e, sys)

    def run_full_pipeline(self, experiment_name="default_run", use_augmentation=False):
        try:
            x_train, x_val, x_test, y_train, y_val, y_test = self.run_data_pipeline()

            model, history = self.run_training_pipeline(
                x_train, y_train, x_val, y_val, use_augmentation=use_augmentation
            )

            evaluator, results = self.run_evaluation_pipeline(model, x_test, y_test, history)

            self.run_tracking_pipeline(experiment_name, use_augmentation, history, results)

            top_pairs = self.run_misclassification_pipeline(evaluator)

            return model, history, results, top_pairs
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = Pipeline()
    model, history, results, top_pairs = pipeline.run_full_pipeline(
        experiment_name="with_augmentation",
        use_augmentation=True
    )

    print(f"\nTEST ACCURACY: {results['test_accuracy']:.4f}")

    print("\nTop Confused Class Pairs:")
    for pair in top_pairs:
        print(pair)
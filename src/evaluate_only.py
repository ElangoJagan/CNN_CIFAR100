from src.exception import CustomException
from src.logger import Logger
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.evaluator import Evaluator
from src.configfile import DataLoaderConfig

from tensorflow.keras.models import load_model
import sys

_logger_obj = Logger('EvaluateOnly')
logger = _logger_obj.get_logger()


def main():
    try:
        # Step 1: Get test data (reuses existing saved raw data, no retrain)
        config = DataLoaderConfig()
        loader = DataLoader()
        raw_data_path = loader.initiate_data_loader()  # skips download since file exists

        preprocessor = Preprocessor(raw_data_path=raw_data_path)
        x_train, x_val, x_test, y_train, y_val, y_test = preprocessor.initiate_preprocessing()
        logger.info("Test data ready")

        # Step 2: Load the already-trained model from disk
        model = load_model('artifacts/models/best_model.keras')
        logger.info("Saved model loaded from artifacts/models/best_model.keras")

        # Step 3: Evaluate on test set
        evaluator = Evaluator(model, x_test, y_test)
        evaluator.evaluate()
        report = evaluator.get_classification_report()
        cm_path = evaluator.plot_confusion_matrix()

        print("\n" + "="*50)
        print(f"TEST ACCURACY: {evaluator.test_accuracy:.4f}")
        print("="*50)
        print("\nClassification Report:\n")
        print(report)
        print(f"\nConfusion matrix saved: {cm_path}")
        print(f"\n{evaluator}")  # uses __str__

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
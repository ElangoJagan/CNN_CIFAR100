from src.exception import CustomException
from src.logger import Logger
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.evaluator import Evaluator
from src.misclassification_analyzer import MisclassificationAnalyzer
from src.configfile import DataLoaderConfig

from tensorflow.keras.models import load_model
import sys

_logger_obj = Logger('EvaluateOnly')
logger = _logger_obj.get_logger()


def main():
    try:
        # Step 1: Test data ready pannuradhu (fast -- retrain illa)
        config = DataLoaderConfig()
        loader = DataLoader()
        raw_data_path = loader.initiate_data_loader()

        preprocessor = Preprocessor(raw_data_path=raw_data_path)
        x_train, x_val, x_test, y_train, y_val, y_test = preprocessor.initiate_preprocessing()
        logger.info("Test data ready")

        # Step 2: Already trained model load pannuradhu
        model = load_model('artifacts/models/best_model.keras')
        logger.info("Saved model loaded")

        # Step 3: Evaluate
        evaluator = Evaluator(model, x_test, y_test)
        evaluator.evaluate()
        report = evaluator.get_classification_report()
        cm_path = evaluator.plot_confusion_matrix()

        print("\n" + "="*50)
        print(f"TEST ACCURACY: {evaluator.test_accuracy:.4f}")
        print("="*50)
        print("\nClassification Report:\n")
        print(report)

        # Step 4: Misclassification analysis
        analyzer = MisclassificationAnalyzer(evaluator.y_test, evaluator.y_pred)
        top_pairs = analyzer.initiate_analysis()

        print("\nTop Confused Class Pairs:")
        for pair in top_pairs:
            print(pair)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
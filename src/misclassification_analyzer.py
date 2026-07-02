from src.exception import CustomException
from src.logger import Logger
import sys

import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import cifar100

_logger_obj = Logger('MisclassificationAnalyzer')
logger = _logger_obj.get_logger()

CIFAR100_CLASS_NAMES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]


class MisclassificationAnalyzer:

    TOP_N = 10
    CLASS_NAMES = CIFAR100_CLASS_NAMES

    def __init__(self, y_test, y_pred):
        logger.info("MisclassificationAnalyzer initialized")
        self.y_test = y_test
        self.y_pred = y_pred
        self.cm = None

    def build_confusion_matrix(self):
        try:
            self.cm = confusion_matrix(self.y_test, self.y_pred)
            logger.info("Confusion matrix built for analysis")
            return self.cm
        except Exception as e:
            raise CustomException(e, sys)

    def get_top_confused_pairs(self):
        try:
            if self.cm is None:
                self.build_confusion_matrix()

            confused_pairs = []
            num_classes = self.cm.shape[0]

            for actual in range(num_classes):
                for predicted in range(num_classes):
                    if actual != predicted and self.cm[actual][predicted] > 0:
                        confused_pairs.append({
                            'actual_class': actual,
                            'predicted_class': predicted,
                            'count': int(self.cm[actual][predicted])
                        })

            confused_pairs = sorted(confused_pairs, key=lambda x: x['count'], reverse=True)
            top_pairs = confused_pairs[:self.TOP_N]
            logger.info(f"Top {self.TOP_N} confused pairs identified")
            return top_pairs
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def get_coarse_labels():
        try:
            (_, y_train_coarse), (_, y_test_coarse) = cifar100.load_data(label_mode='coarse')
            logger.info("Coarse labels loaded for superclass mapping")
            return y_test_coarse.flatten()
        except Exception as e:
            raise CustomException(e, sys)

    def analyze_with_coarse_labels(self, top_pairs):
        try:
            y_test_coarse = self.get_coarse_labels()

            fine_to_coarse = {}
            for i in range(len(self.y_test)):
                fine_to_coarse[self.y_test[i]] = y_test_coarse[i]

            for pair in top_pairs:
                actual_coarse = fine_to_coarse.get(pair['actual_class'])
                predicted_coarse = fine_to_coarse.get(pair['predicted_class'])
                pair['same_superclass'] = bool(actual_coarse == predicted_coarse)
                pair['actual_name'] = self.CLASS_NAMES[pair['actual_class']]
                pair['predicted_name'] = self.CLASS_NAMES[pair['predicted_class']]

            logger.info("Coarse label analysis completed")
            return top_pairs
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_analysis(self):
        try:
            top_pairs = self.get_top_confused_pairs()
            top_pairs = self.analyze_with_coarse_labels(top_pairs)
            logger.info("Misclassification analysis completed")
            return top_pairs
        except Exception as e:
            raise CustomException(e, sys)
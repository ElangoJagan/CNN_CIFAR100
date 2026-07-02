from src.exception import CustomException
from src.logger import Logger
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

_logger_obj = Logger('Evaluator')
logger = _logger_obj.get_logger()   

class Evaluator:
    PLOT_SAVE_PATH = 'artifacts/plots/'
    
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test    
        self.y_pred = None
        self.test_accuracy = None
    
    def predict(self):
        try:
            y_pred_prob = self.model.predict(self.x_test)
            self.y_pred= np.argmax(y_pred_prob, axis = 1)
            logger.info("Prediction completed successfully")
            return self.y_pred
        except Exception as e:
            raise CustomException(e, sys)
    
    def evaluate(self):
        try:
            if self.y_pred is None:
                self.predict()
            
            self.test_accuracy = accuracy_score(self.y_test, self.y_pred)
            logger.info(f"Test Accuracy: {self.test_accuracy:.4f}")
            return self.test_accuracy
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_classification_report(self):  #per-class breakdown
        try:
            if self.y_pred is None:
                self.predict()
            report = classification_report(self.y_test, self.y_pred)
            logger.info("Classification report generated successfully")
            return report
        except Exception as e:
            raise CustomException(e, sys)
    
    def plot_confusion_matrix(self):
        try:
            if self.y_pred is None:
                self.predict()
            
            cm = confusion_matrix(self.y_test, self.y_pred)
            
            plt.figure(figsize=(20,18))
            sns.heatmap(cm, cmap = 'Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            plt.title('Confusion Matrix - CIFAR 100')
            
            save_path = self.PLOT_SAVE_PATH+'confusion_matrix.png'
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
            plt.close()
            return save_path
        except Exception as e:
            raise CustomException(e, sys)
    
    def plot_training_curves(self,history):
        try:
            fig, axes = plt.subplots(1,2, figsize = (14,5))
            
            axes[0].plot(history.history['accuracy'], label = 'Train Accuracy')
            axes[0].plot(history.history['val_accuracy'], label = 'Val Accuracy')
            axes[0].set_title('accuracy vs epochs')
            axes[0].set_xlabel('Epochs')
            axes[0].legend()
            
            axes[1].plot(history.history['loss'], label='Train Loss')
            axes[1].plot(history.history['val_loss'], label='Val Loss')
            axes[1].set_title('Loss over Epochs')
            axes[1].set_xlabel('Epoch')
            axes[1].legend()
            
            save_path = self.PLOT_SAVE_PATH+'training_curves.png'
            plt.savefig(save_path)
            plt.close()
            
            logger.info(f"Training curves saved to {save_path}")
            return save_path
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def __str__(self):
        if self.test_accuracy is not None:
            return f"Evaluator(test_accuracy={self.test_accuracy:.4f})"
        return "Evaluator(not yet evaluated)"
    
    def initiate_evaluation(self, history):
        try:
            self.evaluate()
            report = self.get_classification_report()
            cm_path = self.plot_confusion_matrix()
            curves_path = self.plot_training_curves(history)
            
            logger.info('Evaluation completed successfully')
            return {
            'test_accuracy': self.test_accuracy,
            'report': report,
            'confusion_matrix_path': cm_path,
            'training_curves_path': curves_path
        }
        except Exception as e:
            raise CustomException(e, sys)
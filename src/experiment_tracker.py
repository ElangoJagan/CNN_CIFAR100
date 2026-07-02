from src.exception import CustomException
from src.logger import Logger
import sys
import os 
import csv
from dataclasses import dataclass, asdict
from datetime import datetime


_logger_obj = Logger('ExperimentTracker')
logger = _logger_obj.get_logger()


@dataclass
class ExperimentRecord:
    experiment_name :str
    timestamp:str
    epochs:int
    batch_size:int
    augmentation:bool
    train_accuracy:float
    val_accuracy:float
    test_accuracy:float
    

class ExperimentTracker:
    
    CSV_PATH = 'artifacts/experiments/experiment_log.csv'
    
    def __init__(self):
        logger.info('ExperimentTracker Initialized')
        os.makedirs(os.path.dirname(self.CSV_PATH),exist_ok = True)
    
    def log_experiment(self, experiment_name, epochs, batch_size, augmentation, train_accuracy, val_accuracy, test_accuracy):
        try:
            record = ExperimentRecord(
                experiment_name =  experiment_name,
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                epochs= epochs,
                batch_size = batch_size,
                augmentation = augmentation,
                train_accuracy =round(train_accuracy,4),
                val_accuracy = round(val_accuracy, 4),
                test_accuracy = round(test_accuracy, 4)
            )
            
            file_exist = os.path.exists(self.CSV_PATH)
            
            with open(self.CSV_PATH, mode = 'a', newline = '') as f:
                writer = csv.DictWriter(f, fieldnames = asdict(record).keys())
                
                if not file_exist:
                    writer.writeheader()
                
                writer.writerow(asdict(record))
            
            logger.info(f"Experiment logged: {experiment_name} -- test_accuracy={test_accuracy:.4f}")
            return record
        except Exception as e:
            raise CustomException(e,sys)
    
    def show_all_experiments(self):
        try:
            if not os.path.exists(self.CSV_PATH):
                logger.info('No Experiments logged yet')
                return []
            
            with open(self.CSV_PATH, mode = 'r') as f:
                reader = csv.DictReader(f)
                records = list(reader)
            
            logger.info(f"Loaded {len(records)} experiment records")
            return records
        except Exception as e:
            raise CustomException(e, sys)
                
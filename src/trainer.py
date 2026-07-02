from src.exception import CustomException
from src.logger import Logger
import sys

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.augmentor import Augmentor

_logger_obj = Logger('Trainer')
logger = _logger_obj.get_logger()


class Trainer:

    EPOCHS = 30
    BATCH_SIZE = 64
    PATIENCE = 5
    CHECKPOINT_PATH = 'artifacts/models/best_model.keras'

    def __init__(self, model):
        logger.info("Trainer initialized")
        self.model = model

    def get_callbacks(self):
        try:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=self.PATIENCE,
                restore_best_weights=True
            )

            checkpoint = ModelCheckpoint(
                filepath=self.CHECKPOINT_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )

            logger.info("Callbacks created: EarlyStopping, ModelCheckpoint")
            return [early_stop, checkpoint]
        except Exception as e:
            raise CustomException(e, sys)

    def train(self, x_train, y_train, x_val, y_val, use_augmentation=False):
        try:
            callbacks = self.get_callbacks()

            logger.info(f"Training started -- epochs={self.EPOCHS}, "
                        f"batch_size={self.BATCH_SIZE}, augmentation={use_augmentation}")

            if use_augmentation:
                datagen = Augmentor().build_default()
                datagen.fit(x_train)

                history = self.model.fit(
                    datagen.flow(x_train, y_train, batch_size=self.BATCH_SIZE),
                    validation_data=(x_val, y_val),
                    epochs=self.EPOCHS,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                history = self.model.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=self.EPOCHS,
                    batch_size=self.BATCH_SIZE,
                    callbacks=callbacks,
                    verbose=1
                )

            logger.info("Training completed")
            return history
        except Exception as e:
            raise CustomException(e, sys)

    def train_with_dataset(self, train_ds, val_ds):
        try:
            callbacks = self.get_callbacks()
            logger.info(f"Training started (tf.data pipeline) -- epochs={self.EPOCHS}")

            history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.EPOCHS,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("Training completed")
            return history
        except Exception as e:
            raise CustomException(e, sys)
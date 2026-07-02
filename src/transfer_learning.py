from src.exception import CustomException
from src.logger import Logger
from src.model import CNNModel
import sys

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

_logger_obj = Logger('TransferLearning')
logger = _logger_obj.get_logger()


class TransferModel(CNNModel):

    IMG_SIZE = 96          # MobileNetV2 needs at least 32x32, but performs better upscaled
    DENSE_UNITS = 256
    DROPOUT_RATE = 0.5
    NUM_CLASSES = 100
    LEARNING_RATE = 0.0001   # lower LR -- fine-tuning pretrained weights needs gentler updates
    UNFREEZE_LAYERS = 20     # last N layers of MobileNet to unfreeze for fine-tuning

    def __init__(self):
        logger.info("TransferModel initialized")
        self.model = None
        self.base_model = None

    def build_model(self):
        try:
            inputs = Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3))

            self.base_model = MobileNetV2(
                input_tensor=inputs,
                include_top=False,        # MobileNet's own classifier head remove pannuvom
                weights='imagenet'        # pretrained ImageNet weights load pannuvom
            )

            # Base model layers ellam freeze pannuvom first (pretrained knowledge protect pannuradhukku)
            self.base_model.trainable = False

            x = self.base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(self.DENSE_UNITS, activation='relu')(x)
            x = Dropout(self.DROPOUT_RATE)(x)
            outputs = Dense(self.NUM_CLASSES, activation='softmax')(x)

            self.model = Model(inputs=inputs, outputs=outputs)

            logger.info("Transfer learning model built (base frozen)")
            logger.info(f"Total layers: {len(self.model.layers)}")

            return self.model
        except Exception as e:
            raise CustomException(e, sys)

    def unfreeze_top_layers(self):
        try:
            self.base_model.trainable = True

            # last UNFREEZE_LAYERS mattum trainable ah vachikkuvom, mudhal layers freeze ah irukum
            for layer in self.base_model.layers[:-self.UNFREEZE_LAYERS]:
                layer.trainable = False

            logger.info(f"Unfroze last {self.UNFREEZE_LAYERS} layers of base model for fine-tuning")
        except Exception as e:
            raise CustomException(e, sys)

    def compile_model(self):
        try:
            if self.model is None:
                raise ValueError("Model not built yet. Call build_model() first.")

            self.model.compile(
                optimizer=Adam(learning_rate=self.LEARNING_RATE),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info("Transfer model compiled successfully")
            return self.model
        except Exception as e:
            raise CustomException(e, sys)

    def get_model(self, fine_tune=False):
        try:
            self.build_model()

            if fine_tune:
                self.unfreeze_top_layers()

            self.compile_model()
            return self.model
        except Exception as e:
            raise CustomException(e, sys)
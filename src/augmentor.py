from src.exception import CustomException
from src.logger import Logger
import sys

from tensorflow.keras.preprocessing.image import ImageDataGenerator

_logger_obj = Logger('Augmentor')
logger = _logger_obj.get_logger()


class Augmentor:

    ROTATION_RANGE = 15
    WIDTH_SHIFT_RANGE = 0.1
    HEIGHT_SHIFT_RANGE = 0.1
    HORIZONTAL_FLIP = True
    ZOOM_RANGE = 0.1

    def __init__(self):
        logger.info("Augmentor initialized")
        self.datagen = None
        self.config = {}

    def set_rotation(self, degrees):
        self.config['rotation_range'] = degrees
        return self

    def set_shift(self, width_shift, height_shift):
        self.config['width_shift_range'] = width_shift
        self.config['height_shift_range'] = height_shift
        return self

    def set_flip(self, horizontal_flip):
        self.config['horizontal_flip'] = horizontal_flip
        return self

    def set_zoom(self, zoom_range):
        self.config['zoom_range'] = zoom_range
        return self

    def build(self):
        try:
            self.datagen = ImageDataGenerator(**self.config)
            logger.info(f"ImageDataGenerator built with config: {self.config}")
            return self.datagen
        except Exception as e:
            raise CustomException(e, sys)

    def build_default(self):
        try:
            self.set_rotation(self.ROTATION_RANGE) \
                .set_shift(self.WIDTH_SHIFT_RANGE, self.HEIGHT_SHIFT_RANGE) \
                .set_flip(self.HORIZONTAL_FLIP) \
                .set_zoom(self.ZOOM_RANGE)

            return self.build()
        except Exception as e:
            raise CustomException(e, sys)
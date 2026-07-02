from src.exception import CustomException
from src.logger import Logger
import sys
import os

import numpy as np
import tensorflow as tf
import cv2

_logger_obj = Logger('Interpretability')
logger = _logger_obj.get_logger()


class GradCAM:

    SAVE_PATH = 'artifacts/plots/gradcam/'

    def __init__(self, model, last_conv_layer_name=None):
        logger.info("GradCAM initialized")
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name or self._find_last_conv_layer()

    def _find_last_conv_layer(self):
        try:
            for layer in reversed(self.model.layers):
                if 'conv2d' in layer.name:
                    logger.info(f"Auto-detected last conv layer: {layer.name}")
                    return layer.name
            raise ValueError("No Conv2D layer found in model")
        except Exception as e:
            raise CustomException(e, sys)

    def generate_heatmap(self, img_array, class_index=None):
        try:
            inputs = tf.keras.Input(shape=self.model.input_shape[1:])
            x = inputs
            conv_output_tensor = None

            for layer in self.model.layers:
                x = layer(x)
                if layer.name == self.last_conv_layer_name:
                    conv_output_tensor = x

            grad_model = tf.keras.models.Model(inputs=inputs, outputs=[conv_output_tensor, x])

            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(img_array)
                if class_index is None:
                    class_index = tf.argmax(predictions[0])
                class_channel = predictions[:, class_index]

            grads = tape.gradient(class_channel, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_output = conv_output[0]
            heatmap = conv_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()

            logger.info(f"Heatmap generated for class_index={class_index}")
            return heatmap, int(class_index)
        except Exception as e:
            raise CustomException(e, sys)

    def overlay_heatmap(self, heatmap, original_img, alpha=0.4):
        try:
            heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            original_uint8 = np.uint8(255 * original_img)
            superimposed = cv2.addWeighted(original_uint8, 1 - alpha, heatmap_colored, alpha, 0)

            logger.info("Heatmap overlaid on original image")
            return superimposed
        except Exception as e:
            raise CustomException(e, sys)

    def save_gradcam(self, img_array, original_img, filename, class_index=None):
        try:
            os.makedirs(self.SAVE_PATH, exist_ok=True)

            heatmap, predicted_class = self.generate_heatmap(img_array, class_index)
            superimposed = self.overlay_heatmap(heatmap, original_img)

            save_path = f"{self.SAVE_PATH}{filename}.png"
            cv2.imwrite(save_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))

            logger.info(f"Grad-CAM saved at {save_path} for predicted_class={predicted_class}")
            return save_path, predicted_class
        except Exception as e:
            raise CustomException(e, sys)
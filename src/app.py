from src.exception import CustomException
from src.logger import Logger
import sys
import os

import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image

_logger_obj = Logger('App')
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

app = Flask(__name__, template_folder='../templates', static_folder='../static')

MODEL_PATH = 'artifacts/models/best_model.keras'
UPLOAD_FOLDER = 'static/uploads'
IMG_SIZE = 32
TOP_K = 5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    model = load_model(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    raise CustomException(e, sys)


def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise CustomException(e, sys)


def get_top_predictions(img_array):
    try:
        predictions = model.predict(img_array)[0]
        top_indices = np.argsort(predictions)[::-1][:TOP_K]

        results = []
        for idx in top_indices:
            results.append({
                'class_name': CIFAR100_CLASS_NAMES[idx],
                'confidence': round(float(predictions[idx]) * 100, 2)
            })
        return results
    except Exception as e:
        raise CustomException(e, sys)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', predictions=None, image_path=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('index.html', predictions=None, image_path=None, error="No file uploaded")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', predictions=None, image_path=None, error="No file selected")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        logger.info(f"Image uploaded: {filepath}")

        img_array = preprocess_image(filepath)
        predictions = get_top_predictions(img_array)
        logger.info(f"Top prediction: {predictions[0]['class_name']} ({predictions[0]['confidence']}%)")

        return render_template('index.html', predictions=predictions, image_path=filepath)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(debug=True)
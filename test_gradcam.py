from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.interpretability import GradCAM
from tensorflow.keras.models import load_model
import numpy as np

loader = DataLoader()
raw_path = loader.initiate_data_loader()

preprocessor = Preprocessor(raw_data_path=raw_path)
x_train, x_val, x_test, y_train, y_val, y_test = preprocessor.initiate_preprocessing()

model = load_model('artifacts/models/best_model.keras')
gradcam = GradCAM(model)

sample_img = x_test[0]
img_array = np.expand_dims(sample_img, axis=0)   

save_path, predicted_class = gradcam.save_gradcam(
    img_array=img_array,
    original_img=sample_img,
    filename='sample_0',
    class_index=None   
)

print(f"Saved: {save_path}, Predicted class: {predicted_class}")
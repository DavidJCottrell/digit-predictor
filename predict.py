import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('num_reader.keras')

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L') 
    img = img.resize((28, 28)) 
    img_array = np.array(img) / 255.0
    img_array = 1 - img_array
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    return predicted_digit

predicted_digit = predict_image('test_number.png')
print(f"\nIs it a {predicted_digit}? 😬\n")
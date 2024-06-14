import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image, UnidentifiedImageError
import numpy as np
from tensorflow.keras.models import load_model
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import random
from termcolor import colored

model = load_model('num_reader.keras')

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L') 
    img = img.resize((28, 28)) 
    img_array = np.array(img) / 255.0 # Normalize pixel values
    img_array = 1 - img_array # Invert colors as MNIST has white digit on black background
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = np.expand_dims(img_array, axis=-1) 
    return img_array

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    return model.predict(img_array)

def print_predication(predictions):
    os.system('cls' if os.name == 'nt' else 'clear')

    sorted_predictions = np.argsort(-predictions[0])

    for prediction_index in sorted_predictions:
        print(f"    {colored(prediction_index, 'green', attrs=['bold'])}: {round(predictions[0][prediction_index] * 100, 2)}%")
        if(prediction_index == np.argmax(predictions)):
            print("-------------------")

class MyHandler(FileSystemEventHandler):
    def on_modified(self, _):
        global last_file_change_time
        try:
            predictions = predict_image(f"./digits/test_number.png")
            print_predication(predictions)
        except UnidentifiedImageError:
            print("Error reading image file")

if __name__ == "__main__":
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path='./digits', recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

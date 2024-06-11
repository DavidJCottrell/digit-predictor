import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from datetime import datetime, timedelta

model = load_model('num_reader.keras')

last_file_change_time = datetime.now()

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

def print_predication(predicted_digit):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"\nIs it a {predicted_digit}? 😬\n")

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global last_file_change_time
        # Ignore duplicated events
        if((datetime.now() - last_file_change_time) >= timedelta(milliseconds=100)):
            predicted_digit = predict_image(f"{event.src_path}/test_number.png")
            print_predication(predicted_digit)
            last_file_change_time = datetime.now()

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

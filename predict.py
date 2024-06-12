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

model = load_model('num_reader.keras')

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L') # Convert to grayscale
    img = img.resize((28, 28)) # Resize to 28x28 (matching MNIST)
    img_array = np.array(img) / 255.0 # Normalize pixel values
    img_array = 1 - img_array # Invert colors: MNIST has white digit on black background
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1) # Add channel dimension
    return img_array

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    return predicted_digit

def print_predication(predicted_digit):
    os.system('cls' if os.name == 'nt' else 'clear')

    choice = random.randrange(1, 5)
    if(choice == 1):
        print(f"\nNow that's a nice {predicted_digit}!\n")
    elif(choice == 2):
        print(f"\nMotherfuckin {predicted_digit}\n")
    elif(choice == 3):
        print(f"\nNow would you look at that {predicted_digit}!\n")
    elif(choice == 4):
        print(f"\nI've seen a lot of {predicted_digit}'s, but that's the nicest 👍\n")

class MyHandler(FileSystemEventHandler):
    def on_modified(self, _):
        global last_file_change_time
        try:
            predicted_digit = predict_image(f"./digits/test_number.png")
            print_predication(predicted_digit)
        except UnidentifiedImageError:
            print("Error")

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

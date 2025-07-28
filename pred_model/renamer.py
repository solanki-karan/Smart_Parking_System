import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained digit classification model
model = load_model('digit_classifier_custom.h5')

# Coordinates for hour, minute, and second digits (x1, y1, x2, y2)
digit_coords = [
    (268, 0, 289, 33),  # hour tens
    (289, 0, 310, 33),  # hour units
    (338, 0, 359, 33),  # minute tens
    (359, 0, 380, 33),  # minute units
    (410, 0, 431, 33),  # second tens
    (431, 0, 452, 33),  # second units
]

def predict_digit(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (30, 50))  # match model input size
    gray = gray.astype('float32') / 255.0
    gray = np.expand_dims(gray, axis=(0, -1))
    pred = model.predict(gray, verbose=0)
    return str(np.argmax(pred))

def extract_timestamp_from_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None
    digit_images = [image[y1:y2, x1:x2] for (x1, y1, x2, y2) in digit_coords]
    digits = [predict_digit(d) for d in digit_images]
    return f"{digits[0]}{digits[1]}{digits[2]}{digits[3]}{digits[4]}{digits[5]}"

def rename_all_images_in_folder(folder_path):
    used_names = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            timestamp = extract_timestamp_from_image(img_path)
            if timestamp:
                new_name = f"moment_{timestamp}"
                count = used_names.get(new_name, 0)
                used_names[new_name] = count + 1

                # Add suffix if duplicate timestamp exists
                if count > 0:
                    new_name += f"_{count}"

                new_file_path = os.path.join(folder_path, new_name + ".png")
                os.rename(img_path, new_file_path)
                print(f"Renamed {filename} → {new_name}.png")
            else:
                print(f"Skipped: {filename} (could not read)")

# === USAGE ===
frames_folder = 'pred_model/moments/untitled'  # Change this to your folder path
rename_all_images_in_folder(frames_folder)

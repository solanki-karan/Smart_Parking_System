from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load pretrained YOLOv8 model
model = YOLO('yolov8l.pt')  # Or yolov8m.pt, yolov8l.pt

# Classes to detect: car(2), bus(5), truck(7)
target_classes = [2, 5, 7]
class_names = {2: 'car', 5: 'bus', 7: 'truck'}

# Define polygon as list of (x, y) tuples
polygon = np.array([
    (30, 300),
    (1200, 360),
    (1250, 550),
    (30, 450)
], dtype=np.int32)

# Parent folder containing all subfolders
parent_folder = 'pred_model/moments'
output_log_path = os.path.join(parent_folder, 'vehicle_count_log.txt')

# Open a single log file for all entries
with open(output_log_path, 'w') as log_file:
    # Iterate through each subfolder
    for subfolder in sorted(os.listdir(parent_folder)):
        input_folder = os.path.join(parent_folder, subfolder)
        if not os.path.isdir(input_folder):
            continue

        frame_index = 0
        print(f"🔍 Processing folder: {subfolder}")

        for filename in sorted(os.listdir(input_folder)):
            if frame_index % 4 != 0:
                frame_index += 1
                continue

            if not filename.lower().endswith('.png'):
                continue

            image_path = os.path.join(input_folder, filename)
            img = cv2.imread(image_path)
            if img is None:
                continue

            results = model(img, classes=target_classes)
            count_inside_polygon = 0

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                        count_inside_polygon += 1

                    label = class_names.get(cls, "object")
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.polylines(img, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)

            # Extract timestamp from filename: moment_104501.png → 104501
            time_str = filename.replace('moment_', '').replace('.png', '')

            # Log entry: just the timestamp and vehicle count
            log_file.write(f"{time_str}: {count_inside_polygon} vehicles\n")
            print(f"{filename} → {count_inside_polygon} vehicles")

            frame_index += 1

            # Optional: save annotated frame
            # cv2.imwrite(os.path.join(input_folder, 'annotated_' + filename), img)

print(f"✅ Combined log saved at: {output_log_path}")
cv2.imshow('Last Processed Frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

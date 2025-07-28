# import cv2
# import numpy as np
# from ultralytics import YOLO

# def adjust_gamma(image, gamma=1.5):
#     inv_gamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** inv_gamma) * 255 
#                       for i in np.arange(256)]).astype("uint8")
#     return cv2.LUT(image, table)

# def detect_cars_in_night_image(image_path, gamma_value=1.8, conf_threshold=0.3):
#     # Load image
#     img = cv2.imread(image_path)
#     if img is None:
#         print("Error: Could not load image.")
#         return
    
#     # Brighten image with gamma correction
#     bright_img = adjust_gamma(img, gamma=gamma_value)

#     # Load YOLOv8 pretrained model (YOLOv8n for speed, change to 'yolov8m.pt' or others if needed)
#     model = YOLO('yolov8n.pt')

#     # Run detection on brightened image
#     results = model(bright_img)

#     # Extract detected boxes for 'car' class (class 2 in COCO)
#     car_class_id = [2, 5, 7]
#     annotated_img = bright_img.copy()
    
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])
#             if cls_id in car_class_id and conf > conf_threshold:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = f"Car {conf:.2f}"
#                 # Draw bounding box and label
#                 cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(annotated_img, label, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
#     # Show original and brightened images side-by-side
#     combined = np.hstack((img, annotated_img))
#     cv2.imshow("Original (Left) vs Brightened + YOLOv8 Detection (Right)", combined)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Replace with your night image path
#     image_path = "frames/frame_00500.png"
#     detect_cars_in_night_image(image_path)


from ultralytics import YOLO
import cv2
import numpy as np


# Load pretrained YOLOv8 model
model = YOLO('yolov8l.pt')  # or yolov8m.pt, yolov8l.pt

# Classes to detect: car(2), bus(5), truck(7)
target_classes = [2, 5, 7]
class_names = {2: 'car', 5: 'bus', 7: 'truck'}

# Read input image
img = cv2.imread('ss.png')

# Run prediction filtering for target classes
results = model(img, classes=target_classes)

# Draw bounding boxes with labels
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())  # predicted class id
        label = class_names.get(cls, "object")

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put label text above the box
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Show the result
cv2.imshow('Detected Vehicles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

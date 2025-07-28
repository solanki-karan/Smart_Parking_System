import cv2
import numpy as np
import ast
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

# ---------- Load parking spot polygons from TXT file ----------
def load_parking_spots(file_path):
    spots = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                polygon = ast.literal_eval(line)
                spots.append(polygon)
    return spots

parking_spots = load_parking_spots("polygon_output.txt")

# ---------- Load the input image ----------
image_path = "pred_model/moments/5_jun/moment_132426.png"
image = cv2.imread(image_path)

# ---------- Load YOLOv8x model ----------
model = YOLO("yolov8l.pt")
results = model(image)[0]

# ---------- Get only vehicle detections (car=2, bus=5, truck=7) ----------
vehicle_boxes = [
    box.xyxy[0].tolist()
    for box in results.boxes
    if int(box.cls[0]) in [2, 5, 7]
]

vehicle_centers = [((x1 + x2)/2, (y1 + y2)/2) for (x1, y1, x2, y2) in vehicle_boxes]

# ---------- Check polygon occupancy ----------
empty_spots = [

]
for i, spot in enumerate(parking_spots):
    poly = Polygon(spot)
    occupied = any(poly.contains(Point(center)) for center in vehicle_centers)
    if occupied:
        empty_spots.append(i)

    # Draw polygon
    color = (0, 0, 255) if occupied else (0, 255, 0)
    cv2.polylines(image, [np.array(spot, dtype=np.int32)], isClosed=True, color=color, thickness=2)

    # Label spot
    cx = int(np.mean([pt[0] for pt in spot]))
    cy = int(np.mean([pt[1] for pt in spot]))
    cv2.putText(image, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# ---------- Draw vehicle boxes ----------
for (x1, y1, x2, y2) in vehicle_boxes:
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

# ---------- Show image ----------
cv2.imshow("Occupancy Detection", image)
print("Empty spots:")
print(empty_spots)

cv2.waitKey(0)
cv2.destroyAllWindows()

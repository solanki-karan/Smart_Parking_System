import cv2
import numpy as np
import ast

# ---------- Load parking spot polygons ----------
def load_parking_spots(file_path):
    spots = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                polygon = ast.literal_eval(line)
                spots.append(polygon)
    return spots

# ---------- Input ----------
blueprint_path = "navigation/layout.png"               # Your blueprint image
spots_file_path = "navigation/layout_polygons.txt"                  # Polygon coordinates
empty_spots = [0, 2, 4, 7]                     # Indexes of empty spots

# ---------- Load blueprint and polygons ----------
image = cv2.imread(blueprint_path)
parking_spots = load_parking_spots(spots_file_path)

# ---------- Draw each spot ----------
for i, spot in enumerate(parking_spots):
    pts = np.array(spot, dtype=np.int32)
    is_empty = i in empty_spots
    color = (0, 255, 0) if is_empty else (0, 0, 255)  # Green or Red

    cv2.fillPoly(image, [pts], color)
    cx = int(np.mean([p[0] for p in spot]))
    cy = int(np.mean([p[1] for p in spot]))
    cv2.putText(image, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

# ---------- Show result ----------
cv2.imshow("Blueprint Parking Status", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save to file
# cv2.imwrite("annotated_blueprint.jpg", image)

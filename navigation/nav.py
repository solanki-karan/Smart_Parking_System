import cv2
import numpy as np
import ast
import time
import math

def load_parking_spots(file_path):
    spots = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                polygon = ast.literal_eval(line)
                spots.append(polygon)
    return spots

def spot_center(idx):
    pts = np.array(parking_spots[idx], dtype=np.int32) + margin
    cx = int(np.mean([p[0] for p in pts]))
    cy = int(np.mean([p[1] for p in pts]))
    return (cx, cy)

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

def angle_between_points(p1, p2):
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1]  # inverted y-axis
    return math.degrees(math.atan2(dx, dy)) % 360

def relative_direction(arrow_angle, target_angle):
    diff = (target_angle - arrow_angle) % 360
    if diff <= 30 or diff >= 330:
        return "Straight"
    elif 60 <= diff <= 120:
        return "Right"
    elif 240 <= diff <= 300:
        return "Left"
    elif 150 <= diff <= 210:
        return "Back"
    else:
        return ""

# ---------- CONFIG ----------
blueprint_path = "navigation/layout.png"
spots_file_path = "navigation/layout_polygons.txt"
empty_spots = [0, 2, 4, 7]
margin = 100
car_radius = 15
car_speed = 20  # increased sensitivity
arrow_angle = 270  # start facing right

# ---------- LOAD ----------
image = cv2.imread(blueprint_path)
if image is None:
    print("Error loading blueprint image.")
    exit(1)

parking_spots = load_parking_spots(spots_file_path)

canvas_h = image.shape[0] + 2 * margin
canvas_w = image.shape[1] + 2 * margin
canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
canvas[margin:-margin, margin:-margin] = image

# ---------- DRAW INITIAL SPOTS ----------
for i, spot in enumerate(parking_spots):
    pts = np.array(spot, dtype=np.int32) + margin
    color = (0, 255, 0) if i in empty_spots else (0, 0, 255)
    cv2.fillPoly(canvas, [pts], color)
    cx = int(np.mean([p[0] for p in pts]))
    cy = int(np.mean([p[1] for p in pts]))
    cv2.putText(canvas, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

print(f"Initial empty spots: {empty_spots}")

# ---------- SETUP ----------
car_pos = [margin, canvas_h // 2]
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_color = (0, 0, 0)
car_color = (139, 0, 0)

arrow_icon = np.zeros((40, 40, 4), dtype=np.uint8)
pts_arrow = np.array([[20, 5], [5, 35], [35, 35]], np.int32)
cv2.fillPoly(arrow_icon, [pts_arrow], (0, 0, 255, 255))

target_spot_index = min(empty_spots)
target_center = spot_center(target_spot_index)

reached_spot = None
reach_start_time = None

key_map = {
    82: 'up',
    84: 'down',
    81: 'left',
    83: 'right',
    ord('w'): 'up',
    ord('s'): 'down',
    ord('a'): 'left',
    ord('d'): 'right',
}

print("Use arrow keys or WASD to move. Press ESC to quit.")

while True:
    canvas_copy = canvas.copy()

    # Car Dot
    cv2.circle(canvas_copy, tuple(car_pos), car_radius, car_color, -1)

    # Rotated Arrow
    rotated_arrow = rotate_image(arrow_icon, -arrow_angle)
    x_off = car_pos[0] - rotated_arrow.shape[1] // 2
    y_off = car_pos[1] - rotated_arrow.shape[0] // 2

    for c in range(3):
        alpha = rotated_arrow[:, :, 3] / 255.0
        y1, y2 = y_off, y_off + rotated_arrow.shape[0]
        x1, x2 = x_off, x_off + rotated_arrow.shape[1]

        if 0 <= y1 < canvas_copy.shape[0] and y2 <= canvas_copy.shape[0] and \
           0 <= x1 < canvas_copy.shape[1] and x2 <= canvas_copy.shape[1]:
            canvas_copy[y1:y2, x1:x2, c] = (alpha * rotated_arrow[:, :, c] +
                                            (1 - alpha) * canvas_copy[y1:y2, x1:x2, c])

    # Text Message
    if reached_spot is None:
        text = f"Navigate to spot {target_spot_index + 1}"
    else:
        text = f"Reached spot {reached_spot + 1}!"

    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = (canvas_copy.shape[1] - text_size[0]) // 2
    text_y = margin // 2 + text_size[1] // 2
    cv2.putText(canvas_copy, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    # Direction Label
    if reached_spot is None:
        target_angle = angle_between_points(car_pos, target_center)
        nav_text = relative_direction(arrow_angle, target_angle)
        nav_size, _ = cv2.getTextSize(nav_text, font, 1, 2)
        nav_x = (canvas_copy.shape[1] - nav_size[0]) // 2
        nav_y = text_y + nav_size[1] + 10
        cv2.putText(canvas_copy, nav_text, (nav_x, nav_y), font, 1, (0, 0, 0), 2)

    # Movement
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break
    if key in key_map:
        move_dir = key_map[key]
        if move_dir == 'up':
            car_pos[1] = max(margin, car_pos[1] - car_speed)
            arrow_angle = 0
        elif move_dir == 'down':
            car_pos[1] = min(canvas_copy.shape[0] - margin, car_pos[1] + car_speed)
            arrow_angle = 180
        elif move_dir == 'left':
            car_pos[0] = max(margin, car_pos[0] - car_speed)
            arrow_angle = 270
        elif move_dir == 'right':
            car_pos[0] = min(canvas_copy.shape[1] - margin, car_pos[0] + car_speed)
            arrow_angle = 90

    # ---------- Reach Logic ----------
    if reached_spot is None:
        for idx in empty_spots:
            if dist(car_pos, spot_center(idx)) <= 40:
                reached_spot = idx
                reach_start_time = time.time()
                break
    elif time.time() - reach_start_time >= 2 and reached_spot in empty_spots:
        # 2 seconds passed -> mark as occupied
        empty_spots.remove(reached_spot)
        pts = np.array(parking_spots[reached_spot], dtype=np.int32) + margin
        cv2.fillPoly(canvas, [pts], (0, 0, 255))
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(canvas, str(reached_spot + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        print(f"\nUpdated Empty Spots: {empty_spots}")
        cv2.imshow("Parking Navigation", canvas)
        cv2.waitKey(1000)
        print("Navigation ended.")
        break

    cv2.imshow("Parking Navigation", canvas_copy)

cv2.destroyAllWindows()

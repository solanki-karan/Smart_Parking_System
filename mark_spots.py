import cv2

# Globals
drawing = False
points = []

def click_event(event, x, y, flags, param):
    global points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

        # Draw circle on click
        cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

        # Draw polygon lines if there's more than 1 point
        if len(points) > 1:
            cv2.line(image, points[-2], points[-1], (255, 0, 0), 2)

        cv2.imshow("Draw Parking Polygon", image)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Close the polygon
        if len(points) > 2:
            cv2.line(image, points[-1], points[0], (0, 0, 255), 2)
            cv2.imshow("Draw Parking Polygon", image)
            print("\nPolygon complete.")
            print("Final Points:", points)

            # Save or process points here
            with open("layout_polygons.txt", "a") as f:
                f.write(str(points) + "\n")

            # Reset for next polygon
            points = []

# Change to your image path
image_path = "navigation/layout.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("❌ Image not found. Check the path.")

cv2.imshow("Draw Parking Polygon", image)
cv2.setMouseCallback("Draw Parking Polygon", click_event)

print("🔰 Left click to select polygon corners.")
print("🟥 Right click to finish and save one polygon.")

cv2.waitKey(0)
cv2.destroyAllWindows()

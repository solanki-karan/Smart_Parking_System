import cv2

# Load image
img = cv2.imread('pred_model/moments/7_jun/moment_100111.png')  # Change this to your image path
if img is None:
    print("Failed to load image.")
    exit()

# Create a copy for display
display_img = img.copy()

# Mouse callback function
def show_pixel_values(event, x, y, flags, param):
    global display_img
    if event == cv2.EVENT_MOUSEMOVE:
        display_img = img.copy()  # Reset image on every move
        pixel = img[y, x]
        text = f"({x},{y}) - BGR: {pixel.tolist()}"
        cv2.putText(display_img, text, (10, img.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", show_pixel_values)

while True:
    cv2.imshow("Image", display_img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cv2.destroyAllWindows()

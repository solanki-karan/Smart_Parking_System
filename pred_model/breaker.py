import cv2
import os

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Done! Extracted {frame_count} frames to '{output_folder}'.")

# === USAGE ===
video_file = 'pred_model/videos/F0090D403175_20250607204354406_cam0.mov'
output_dir = 'pred_model/moments/5_jun'

extract_frames(video_file, output_dir)

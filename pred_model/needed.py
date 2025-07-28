import os

# Folder containing the images
folder_path = 'pred_model/moments/7_jun/daytime'  # ← change this if needed

# Rename matching files
for filename in os.listdir(folder_path):
    if filename.startswith('moment_19') and filename.endswith('.png'):
        new_filename = 'moment_09' + filename[9:]  # Replace 19 with 09
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_filename)
        os.rename(src, dst)
        print(f"Renamed: {filename} → {new_filename}")

print("\n✅ All matching files renamed.")

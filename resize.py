import cv2
import os

# Path to your images folder

image_folder = r"C:\Project\YOLOv8-multi-task-main\ultralytics\samples"
output_folder = r"C:\Project\YOLOv8-multi-task-main\ultralytics\samples_resized"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Resize images
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    output_path = os.path.join(output_folder, img_name)

    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping corrupt image: {img_name}")
        continue

    # Resize to (720, 1280)
    resized_img = cv2.resize(img, (1280, 720))

    # Save resized image
    cv2.imwrite(output_path, resized_img)
    print(f"Resized and saved: {output_path}")

print("All images resized successfully!")

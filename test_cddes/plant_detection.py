from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

# Load the trained YOLOv8 model
model = YOLO("models/plant-leaf-detection-and-classification.pt")

# Path to your image (watch for spaces in folder names!)
# image_path = "static/uploads/03bb3aec-d570-4d25-afdb-732fc24ebc58_crop_IMG_20250717_132636026.jpg"
image_path = "static/uploads/ef3e39bd-a89c-4e90-b8cd-82c55b370e55_crop_IMG_20250718_115750277.jpg"

# Run inference
results = model(image_path)

# Show detection on the image (OpenCV format to display in Colab)
result_img = results[0].plot()  # Plot result on image array (numpy)

# Convert BGR to RGB for display
result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

# Display in Colab
plt.figure(figsize=(10, 8))
plt.imshow(result_img_rgb)
plt.axis('off')
plt.title("Detected Seedling")
plt.show()

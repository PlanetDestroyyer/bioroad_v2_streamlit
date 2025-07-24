from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from config import LEAF_COUNTER_MODEL

def classify_leaf_color(r, g, b):
    if g > r and g > b and g > 100:
        return "Healthy Green"
    elif r > g and r > b and r > 100:
        return "Brown/Red"
    elif r > 150 and g > 150 and b < 100:
        return "Yellowing"
    else:
        return "Uncertain"

def analyze_leaf_colors(image_path, model_path=LEAF_COUNTER_MODEL, conf_threshold=0.15):
    model = YOLO(model_path)
    results = model.predict(image_path, conf=conf_threshold)
    result = results[0]

    num_leaves = len(result.boxes)
    leaf_colors = []

    original_image = cv2.imread(result.path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    for box in result.boxes:
        
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        
        leaf_crop = original_image[y1:y2, x1:x2]

        
        avg_color = leaf_crop.mean(axis=(0, 1))  
        avg_r, avg_g, avg_b = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])

        
        color_label = classify_leaf_color(avg_r, avg_g, avg_b)

        
        leaf_colors.append(color_label)

    return num_leaves, leaf_colors


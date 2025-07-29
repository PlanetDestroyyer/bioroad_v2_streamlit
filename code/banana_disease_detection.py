import cv2
import numpy as np
import onnxruntime as ort
import json
import os
from inference import get_model
import supervision as sv
import cv2
import numpy as np
import os


def predict_banana_disease(img_path: str, model_path: str = "models/banana_disease_prediction.onnx", label_path: str = "models/class_labels.json") -> str:

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")


    with open(label_path, "r") as f:
        class_labels = json.load(f)


    img = cv2.imread(img_path)
    img = cv2.resize(img, (384, 384))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: img})

    pred_index = int(np.argmax(outputs[0]))
    predicted_label = class_labels[pred_index]
    return predicted_label


def predict_banana_disease_yolo(image_path: str, save_path: str) -> str:
    # Load image
    image = cv2.imread(image_path)

    # Load Roboflow model
    model = get_model(model_id="banana-disease-fomq1-ye97j/1", api_key="155M6h2yp7SRYfYOw9ud")

    # Run inference
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)

    # Shrink bounding boxes to 20%
    scaled_boxes = []
    for box in detections.xyxy:
        x_min, y_min, x_max, y_max = box
        w = x_max - x_min
        h = y_max - y_min
        new_w = w * 0.7
        new_h = h * 0.7
        cx = x_min + w / 2
        cy = y_min + h / 2
        new_x_min = cx - new_w / 2
        new_y_min = cy - new_h / 2
        new_x_max = cx + new_w / 2
        new_y_max = cy + new_h / 2
        scaled_boxes.append([new_x_min, new_y_min, new_x_max, new_y_max])
    detections.xyxy = np.array(scaled_boxes)

    # Labels
    labels = [
        f"{pred.class_name} {pred.confidence:.2f}"
        for pred in results.predictions
    ]

    # Annotate
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # Save annotated image
    cv2.imwrite(save_path, annotated_image)

    # Return first detected class (or 'Unknown')
    return results.predictions[0].class_name if results.predictions else "Unknown"


def predict_banana_disease_yolo(image_path: str, save_path: str) -> list:
    # Load image
    image = cv2.imread(image_path)

    # Load Roboflow model
    model = get_model(model_id="banana-disease-fomq1-ye97j/1", api_key="155M6h2yp7SRYfYOw9ud")

    # Run inference
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)

    # Shrink bounding boxes to 70%
    scaled_boxes = []
    for box in detections.xyxy:
        x_min, y_min, x_max, y_max = box
        w = x_max - x_min
        h = y_max - y_min
        new_w = w * 0.7
        new_h = h * 0.7
        cx = x_min + w / 2
        cy = y_min + h / 2
        new_x_min = cx - new_w / 2
        new_y_min = cy - new_h / 2
        new_x_max = cx + new_w / 2
        new_y_max = cy + new_h / 2
        scaled_boxes.append([new_x_min, new_y_min, new_x_max, new_y_max])
    detections.xyxy = np.array(scaled_boxes)

    # Labels
    labels = [
        f"{pred.class_name} {pred.confidence:.2f}"
        for pred in results.predictions
    ]

    # Annotate
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # Save annotated image
    cv2.imwrite(save_path, annotated_image)

    # Return list of detected class names
    class_names = [pred.class_name for pred in results.predictions]
    return class_names if class_names else ["Unknown"]


if __name__ == "__main__":
    pass
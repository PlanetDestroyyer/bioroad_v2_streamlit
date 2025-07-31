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
    model = get_model(model_id="not_proper_class-1ogfj/2", api_key="UTarhGDr7xOTmafr1t2M")

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


# def predict_banana_disease_yolo(image_path: str, save_path: str) -> list:
#     # Class mapping for mislabeled numbers
#     CLASS_NAME_FIXES = {
#         "0": "bacterial soft rot",
#         "1": "banana aphids",
#         "3": "bract mosaic virus",
#         "5": "cordana",
#         "9": "fusarium wilt",
#         "10": "healthy"
#     }

#     # Load image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Failed to load image at {image_path}")

#     # Load Roboflow model
#     model = get_model(model_id="banana-disease-fomq1-ye97j/1", api_key="155M6h2yp7SRYfYOw9ud")

#     # Run inference
#     results = model.infer(image)[0]
#     print("Inference results:", results)  # Debug
#     detections = sv.Detections.from_inference(results)
#     print("detections.xyxy shape:", detections.xyxy.shape)  # Debug

#     # Shrink bounding boxes to 70%
#     scaled_boxes = []
#     if detections.xyxy.size > 0:  # Check if there are any detections
#         if detections.xyxy.ndim == 1:  # Handle single detection
#             detections.xyxy = detections.xyxy.reshape(1, -1)
#         for box in detections.xyxy:
#             x_min, y_min, x_max, y_max = box
#             w = x_max - x_min
#             h = y_max - y_min
#             new_w = w * 0.7
#             new_h = h * 0.7
#             cx = x_min + w / 2
#             cy = y_min + h / 2
#             new_x_min = cx - new_w / 2
#             new_y_min = cy - new_h / 2
#             new_x_max = cx + new_w / 2
#             new_y_max = cy + new_h / 2
#             scaled_boxes.append([new_x_min, new_y_min, new_x_max, new_y_max])
#         detections.xyxy = np.array(scaled_boxes)
#     else:
#         print("No detections found in the image.")

#     # Fix mislabeled class names if necessary
#     fixed_predictions = []
#     if results.predictions:
#         for pred in results.predictions:
#             class_name = str(pred.class_name)
#             fixed_class = CLASS_NAME_FIXES.get(class_name, class_name)
#             pred.class_name = fixed_class
#             fixed_predictions.append(f"{fixed_class} {pred.confidence:.2f}")
#     else:
#         fixed_predictions = ["No detections"]

#     # Annotate
#     bounding_box_annotator = sv.BoxAnnotator()
#     label_annotator = sv.LabelAnnotator()
#     if detections.xyxy.size > 0:
#         annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
#         annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=fixed_predictions)
#     else:
#         annotated_image = image  # Use original image if no detections

#     # Save annotated image
#     cv2.imwrite(save_path, annotated_image)

#     # Return list of fixed class names
#     class_names = [str(CLASS_NAME_FIXES.get(str(pred.class_name), pred.class_name)) for pred in results.predictions]
#     return class_names if class_names else ["No detections"]


def predict_banana_disease(image_path, save_path):
    """
    Predict banana disease using YOLO model and save annotated image.

    Args:
        image_path (str): Path to the input image.
        model: Loaded YOLO model.
        save_path (str): Path to save the output annotated image.

    Returns:
        list: List of predicted class names or ["Unknown"] if none.
    """
    image = cv2.imread(image_path)
    model = get_model(model_id="not_proper_class-1ogfj/2", api_key="UTarhGDr7xOTmafr1t2M")
    results = model(image)[0]
    detections = sv.Detections.from_inference(results)

    if len(detections.xyxy) == 0:
        cv2.imwrite(save_path, image)
        return ["Unknown"]

    # Shrink bounding boxes to 70%
    scaled_boxes = []
    for box in detections.xyxy:
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        center_x = x_min + width / 2
        center_y = y_min + height / 2
        new_width = width * 0.7
        new_height = height * 0.7
        new_x_min = center_x - new_width / 2
        new_y_min = center_y - new_height / 2
        new_x_max = center_x + new_width / 2
        new_y_max = center_y + new_height / 2
        scaled_boxes.append([new_x_min, new_y_min, new_x_max, new_y_max])

    detections.xyxy = np.array(scaled_boxes)

    # Annotate detections
    box_annotator = sv.BoxAnnotator()
    labels = [results.names[int(class_id)] for class_id in detections.class_id]
    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    cv2.imwrite(save_path, annotated_image)
    return labels


if __name__ == "__main__":
    print(predict_banana_disease_yolo('/home/x/Downloads/black_sigota.jpg','output.jpg'))
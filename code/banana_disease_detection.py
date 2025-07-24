import cv2
import numpy as np
import onnxruntime as ort
import json
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


if __name__ == "__main__":
    pass
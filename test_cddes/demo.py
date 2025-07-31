from ultralytics import YOLO

model = YOLO('models/yolov8n.pt')

model.predict(source='/home/x/Downloads/sample_testin_2.jpg', show=True, save=True)
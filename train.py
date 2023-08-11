import os
# os.environ["OMP_NUM_THREADS"]='2'

from ultralytics import YOLO
# Load a model
model = YOLO('ultralytics/models/v8/yolov8-lite-t-pose.yaml')  # build a new model from YAML
model = YOLO('yolov8-lite-t.pt')  # load a pretrained model (recommended for training)  

# Train the model
model.train(data='v8_plate.yaml', epochs=100, imgsz=320, batch=16, device=[0])

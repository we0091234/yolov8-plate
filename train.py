import os
# os.environ["OMP_NUM_THREADS"]='2'

from ultralytics import YOLO
# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)  

# Train the model
model.train(data='/mnt/mydisk/xiaolei/code/plate/plate_detect/ultralytics-main/ultralytics/cfg/datasets/plate.yaml', epochs=120, imgsz=640, batch=32, device=[0])
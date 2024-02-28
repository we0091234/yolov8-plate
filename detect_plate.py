from ultralytics import YOLO

# 加载预训练的YOLOv8n模型
model = YOLO('runs/detect/train2/weights/best.pt')

# 在'bus.jpg'上运行推理，并附加参数
model.predict('/mnt/mydisk/xiaolei/code/plate/plate_detect/Chinese_license_plate_detection_recognition/imgs/double_yellow.jpg', save=True, imgsz=320, conf=0.5)
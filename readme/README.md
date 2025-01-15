### **车牌检测训练**

1. **下载数据集：**  数据集可以添加vx:we0091234 （注明来意）获取 收费30 介意勿扰    数据从CCPD数据集中选取的一部分，也有自己收集的一部分并转换的
   数据集格式为yolo格式：

   ```
   label x y w h  
   ```
2. **修改ultralytics/datasets/yolov8-plate.yaml    train和val路径,换成你的数据路径**

   ```
   train: /mnt/mydisk/xiaolei/plate_detect/new_train_data # train images (relative to 'path') 4 images
   val: /mnt/mydisk/xiaolei/plate_detect/new_val_data # val images (relative to 'path') 4 images

   # Classes for DOTA 1.0
   names:
   0: single
   1: double

   ```
3. **训练**

   ```
   yolo task=detect mode=train model=yolov8s.yaml  data=./ultralytics/cfg/datasets/plate.yaml epochs=120 batch=32 imgsz=640 pretrained=False optimizer=SGD 
   ```

   结果存在run文件夹中

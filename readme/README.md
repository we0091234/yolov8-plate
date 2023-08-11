### **车牌检测训练**

1. **下载数据集：**  [datasets](https://pan.baidu.com/s/1xa6zvOGjU02j8_lqHGVf0A) 提取码：pi6c     数据从CCPD和CRPD数据集中选取并转换的
   数据集格式为yolo格式：

   ```
   label x y w h  pt1x pt1y pt2x pt2y pt3x pt3y pt4x pt4y
   ```

   关键点依次是（左上，右上，右下，左下）
   坐标都是经过归一化，x,y是中心点除以图片宽高，w,h是框的宽高除以图片宽高，ptx，pty是关键点坐标除以宽高

   **自己的数据集**可以通过lablme 软件,create polygons标注车牌四个点即可，然后通过json2yolo.py 将数据集转为yolo格式，即可训练
2. **修改ultralytics/datasets/yolov8-plate.yaml    train和val路径,换成你的数据路径**

   ```
   train: /mnt/Gu/trainData/CCPD_TRAIN
   val: /mnt/Gu/trainData/CCPD_VAL
   test:  # test images (optional)

   # Keypoints
   kpt_shape: [4, 2]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
   flip_idx:  [1, 0, 3, 2] 

   # Classes
   names:
     0: single
     1: double

   ```
3. **训练
   修改train.py**

   ```
   import os
   # os.environ["OMP_NUM_THREADS"]='2'

   from ultralytics import YOLO
   # Load a model
   model = YOLO('ultralytics/models/v8/yolov8-lite-t-pose.yaml')  # build a new model from YAML
   model = YOLO('yolov8-lite-t.pt')  # load a pretrained model (recommended for training)  #预训练模型可以从yolov8-face这个repo里面找

   # Train the model
   model.train(data='yolov8-plate.yaml', epochs=100, imgsz=320, batch=16, device=[0])

   ```

   运行train.py  结果存在run文件夹中

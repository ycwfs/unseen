from ultralytics import YOLO
import thop  # for FLOPs computation
import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy
model = YOLO('/data1/code/competition/tianchi/unseen/yolov12/yolos_rgb_1280/train/weights/best.pt')

breakpoint()
model.info(detailed=True, verbose=True, imgsz=[1280,723])
results = model.train(
  data='/data1/code/competition/tianchi/unseen/yolov12/ultralytics/cfg/datasets/unseen_augment.yaml',
  epochs=50, 
  batch=48, 
  # cache=True,
  imgsz=[1280,723],
  degrees=1,
  hsv_h=0.015,
  hsv_s=0.1,
  hsv_v=0.3, # lightness
  translate=0.1, # 边缘，部分物体 Translates the image horizontally and vertically by a fraction of the image size, aiding in learning to detect partially visible objects.
  flipud=0, # 上下翻转在这个场景不太适合 Flips the image upside down with the specified probability, increasing the data variability without affecting the object's characteristics.
  fliplr=0.8,
  # shear=2, # 图像四个角的变形 Shears the image by a specified angle, distorting the image while preserving the object's structure, useful for learning object orientation.
  scale=0.1,  # 0.9 ~ 1.1 Scales the image by a gain factor, simulating objects at different distances from the camera.
  mosaic=0.2, #	目标很多，拼接出来训练不一定好 这种增强虽然好，但是由于对裁剪拼接的数据进行了训练。它会破坏检测的完整性。也就是说，如果你的检测画面中存在目标的一小部分，它也会检测出来。有时候，可能我们并不想这样。拿检测汽车来说，如果你希望只检测出完整的汽车，那么mosaic这个开关要关掉。Combines four training images into one, simulating different scene compositions and object interactions. Highly effective for complex scene understanding.
  mixup=0.1,  # 混合两个图片，标签 S:0.05; M:0.15; L:0.15; X:0.2 Blends two images and their labels, creating a composite image. Enhances the model's ability to generalize by introducing label noise and visual variability.
  copy_paste=0.0,  # S:0.15; M:0.4; L:0.5; X:0.6
  device="0",
  workers=20,
  plots=True,
  project='yolol_rgb_1280_continue_augment_training'
)
metrics = model.val()

# 电三轮识别成卡车； 人

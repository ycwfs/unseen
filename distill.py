import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller
# from ultralytics.models.yolo.segment.distill import SegmentationDistiller
# from ultralytics.models.yolo.pose.distill import PoseDistiller
# from ultralytics.models.yolo.obb.distill import OBBDistiller


if __name__ == '__main__':
    param_dict = {
        # origin
        'model': '/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/yolos_rgb_1280/prune_1280/yolov12-finetune/weights/best.pt',
        'data':'/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/ultralytics/cfg/datasets/unseen.yaml',
        'imgsz': [736, 1280],
        'epochs': 300,
        'batch': 52,
        'workers': 8,
        # 'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 20,
        # 'amp': False, # 如果蒸馏损失为nan，请把amp设置为False
        'project':'/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/yolos_rgb_1280',
        'name':'distill_xs_1280_all_halffl_4heads',
        'hsv_h': 0.015,
        'hsv_s': 0.5,
        'hsv_v': 0.6, # lightness
        'translate': 0.0, # 边缘，部分物体 Translates the image horizontally and vertically by a fraction of the image size, aiding in learning to detect partially visible objects.
        'flipud': 0, # 上下翻转在这个场景不太适合 Flips the image upside down with the specified probability, increasing the data variability without affecting the object's characteristics.
        'fliplr': 0.8,
        # shear=2, # 图像四个角的变形 Shears the image by a specified angle, distorting the image while preserving the object's structure, useful for learning object orientation.
        "scale": 0,  # 0.9 ~ 1.1 Scales the image by a gain factor, simulating objects at different distances from the camera.
        "mosaic": 0.6, #	目标很多，拼接出来训练不一定好 这种增强虽然好，但是由于对裁剪拼接的数据进行了训练。它会破坏检测的完整性。也就是说，如果你的检测画面中存在目标的一小部分，它也会检测出来。有时候，可能我们并不想这样。拿检测汽车来说，如果你希望只检测出完整的汽车，那么mosaic这个开关要关掉。Combines four training images into one, simulating different scene compositions and object interactions. Highly effective for complex scene understanding.
        "mixup":0,  # 混合两个图片，标签 S:0.05; M:0.15; L:0.15; X:0.2 Blends two images and their labels, creating a composite image. Enhances the model's ability to generalize by introducing label noise and visual variability.
        "copy_paste": 0.0,  # S:0.15; M:0.4; L:0.5; X:0.6
        "augment": True,
        "ir": False,
        "save_period": 20,
        # distill
        'prune_model': True,
        'teacher_weights': '/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/ckpt/yolox-best.pt',
        'teacher_cfg': '/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/ultralytics/cfg/models/v12/yolov12x.yaml',
        'kd_loss_type': 'all',
        'kd_loss_decay': 'linear',
        
        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 1.0,
        
        'teacher_kd_layers': '11,14,17,20',
        'student_kd_layers': '11,14,17,20',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 0.25
        # Distill settings
        # prune_model: True
        # teacher_weights:
        # teacher_cfg:
        # kd_loss_type: logical # 'logical', 'feature', 'all'
        # kd_loss_decay: constant # 'cosine', 'linear', 'cosine_epoch', 'linear_epoch', 'constant'

        # # logical distillation settings
        # logical_loss_type: l2 # 'l1', 'l2', 'BCKD'
        # logical_loss_ratio: 1.0

        # # feature distillation settings
        # teacher_kd_layers: 15,18,21
        # student_kd_layers: 15,18,21
        # feature_loss_type: cwd # 'mimic', 'mgd', 'cwd', 'chsim', 'sp'
        # feature_loss_ratio: 1.0
    }
    
    
    model = DetectionDistiller(overrides=param_dict)
    # model = SegmentationDistiller(overrides=param_dict)
    # model = PoseDistiller(overrides=param_dict)
    # model = OBBDistiller(overrides=param_dict)
    model.distill()
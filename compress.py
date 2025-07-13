import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.compress import DetectionCompressor, DetectionFinetune
# from ultralytics.models.yolo.segment.compress import SegmentationCompressor, SegmentationFinetune
# from ultralytics.models.yolo.pose.compress import PoseCompressor, PoseFinetune
# from ultralytics.models.yolo.obb.compress import OBBCompressor, OBBFinetune

def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = DetectionCompressor(overrides=param_dict)
    # compressor = SegmentationCompressor(overrides=param_dict)
    # compressor = PoseCompressor(overrides=param_dict)
    # compressor = OBBCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = DetectionFinetune(overrides=param_dict)
    # trainer = SegmentationFinetune(overrides=param_dict)
    # trainer = PoseFinetune(overrides=param_dict)
    # trainer = OBBFinetune(overrides=param_dict)
    trainer.train()

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': '/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/yolos_rgb_1280/train/weights/best.pt',
        'data':'/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/ultralytics/cfg/datasets/unseen.yaml',
        'imgsz': [1280,723],
        'epochs': 200,
        'batch': 16,
        'workers': 8,
        'cache': False,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 0,
        'project': '/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/yolos_rgb_1280/prune_1280',
        'name': 'yolov12',
        'degrees': 1,
        'hsv_h': 0.015,
        'hsv_s': 0.1,
        'hsv_v': 0.3, # lightness
        'translate': 0.1, # 边缘，部分物体 Translates the image horizontally and vertically by a fraction of the image size, aiding in learning to detect partially visible objects.
        'flipud': 0, # 上下翻转在这个场景不太适合 Flips the image upside down with the specified probability, increasing the data variability without affecting the object's characteristics.
        'fliplr': 0.8,
        # shear=2, # 图像四个角的变形 Shears the image by a specified angle, distorting the image while preserving the object's structure, useful for learning object orientation.
        'scale': 0.1,  # 0.9 ~ 1.1 Scales the image by a gain factor, simulating objects at different distances from the camera.
        'mosaic': 0.2, #	目标很多，拼接出来训练不一定好 这种增强虽然好，但是由于对裁剪拼接的数据进行了训练。它会破坏检测的完整性。也就是说，如果你的检测画面中存在目标的一小部分，它也会检测出来。有时候，可能我们并不想这样。拿检测汽车来说，如果你希望只检测出完整的汽车，那么mosaic这个开关要关掉。Combines four training images into one, simulating different scene compositions and object interactions. Highly effective for complex scene understanding.
        'mixup': 0.1,  # 混合两个图片，标签 S:0.05; M:0.15; L:0.15; X:0.2 Blends two images and their labels, creating a composite image. Enhances the model's ability to generalize by introducing label noise and visual variability.
        'copy_paste': 0.0,  # S:0.15; M:0.4; L:0.5; X:0.6
        # 'amp':False,
        
        # prune
        'prune_method': 'lamp',
        'global_pruning': True,
        'speed_up': 2.3,
        'reg': 0.0005,
        'sl_epochs': 500,
        'sl_hyp': '/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None,
    }
    
    prune_model_path = compress(copy.deepcopy(param_dict))
    # prune_model_path = '/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/yolos_rgb_1280/prune_1280/yolov12-prune/weights/prune.pt'
    print(f'prune_model_path: {prune_model_path}')
    finetune(copy.deepcopy(param_dict), prune_model_path)

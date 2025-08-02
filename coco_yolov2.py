import os
import json
from PIL import Image
from datetime import datetime
# ----------------------------
# 路径配置
# ----------------------------
# 获取当前时间并格式化为字符串，比如：20250604_153045
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
coco_pred_json = "/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/result/coco_predictions_s_20250726_120134.json"  # 你的预测结果
image_dir = "/data1/wangqiurui/code/ossutil-v1.7.19-linux-amd64/rgb_unseen/val/images"                   # 图像目录（用于读取图像宽高）
output_label_dir = f"./result/yolo_preds_{timestamp}"           # 输出目录

os.makedirs(output_label_dir, exist_ok=True)

# ----------------------------
# 加载 COCO 格式预测结果
# ----------------------------
with open(coco_pred_json, "r") as f:
    coco_preds = json.load(f)

# ----------------------------
# 收集每张图像的预测框
# ----------------------------
image_pred_map = {}
for pred in coco_preds:
    image_name = pred["image_id"]  # 👈 image_id 是文件名（含后缀）
    if image_name not in image_pred_map:
        image_pred_map[image_name] = []
    image_pred_map[image_name].append(pred)

# ----------------------------
# 写入 YOLO 格式标签文件
# ----------------------------
for image_name, preds in image_pred_map.items():
    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        print(f" 图像文件不存在: {image_name}，跳过。")
        continue

    im = Image.open(image_path)
    w, h = im.size

    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(output_label_dir, label_name)

    with open(label_path, "w") as f:
        for pred in preds:
            x, y, box_w, box_h = pred["bbox"]
            class_id = pred["category_id"]  # 或者 class_id = pred["category_id"] - 1
            conf = pred["score"]
            # # 置信区间映射
            #conf = 0.75 * conf + 0.25
            x_center = (x + box_w / 2) / w
            y_center = (y + box_h / 2) / h
            width = box_w / w
            height = box_h / h
            #f.write(f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf} {class_id}\n")
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf}\n")

print(f" 已完成转换, YOLO 标签保存在：{output_label_dir}")

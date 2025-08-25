import os
import json
from PIL import Image
from tqdm import tqdm
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from datetime import datetime
# ----------------------------
# 配置参数
# ----------------------------

# [720x1280]
# 获取当前时间并格式化为字符串，比如：20250604_153045
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
image_dir = "/data1/code/ossutil-v1.7.19-linux-amd64/rgb_unseen/val/images"
output_json = f"./result/coco_predictions_s_{timestamp}.json"
slice_height = 360  #使用768比512效果要差一个点左右
slice_width = 640
overlap_ratio = 0   #使用0.2效果比0.5,0.1好

# ----------------------------
# 初始化模型（根据你的模型路径）
# ----------------------------
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path="/data1/code/competition/tianchi/unseen/yolov12/yolos_rgb_1280/distill_ls_1280_all_halffl_4heads/weights/best.pt",
    confidence_threshold=0.25,  #用0.2有效果，比0.25好，尺寸为1408训练的情况下。
    device="cuda:1",
)

# ----------------------------
# 推理并收集预测（添加 tqdm 进度条）
# ----------------------------
predictions = []

image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

for filename in tqdm(image_files, desc="Processing images", unit="img"):
    image_path = os.path.join(image_dir, filename)

    # 推理
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
    )

    # 提取 COCO 风格预测
    preds = result.to_coco_predictions(image_id=filename)
    for pred in preds:
        predictions.append({
            "image_id": pred["image_id"],
            "category_id": pred["category_id"],
            "bbox": pred["bbox"],
            "score": pred["score"]
        })

# ----------------------------
# 保存为 COCO JSON
# ----------------------------
with open(output_json, "w") as f:
    json.dump(predictions, f)

print(f"\n 已保存 {len(predictions)} 条预测结果到：{output_json}")

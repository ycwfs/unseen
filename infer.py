from ultralytics import YOLO
from tqdm import tqdm

# Load a pretrained YOLO11n model
model = YOLO("/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/yolos_rgb_1280/prune_1280/yolov12-finetune/weights/best.pt")
# Print the model summary
model.info(detailed=True, verbose=True, imgsz=[1280, 723])
breakpoint()
# Run inference on 'bus.jpg' with arguments
res = model.predict("/data1/wangqiurui/code/ossutil-v1.7.19-linux-amd64/rgb_unseen/val/images/", 
                    save=True, imgsz=1280, conf=0.25, stream=True, device="cuda:0", line_width=1, 
                    save_txt=True, save_conf=True, project='yolos_rgb_1280', name='prune_809_1280_res')
for r in res:
    print(r.summary())
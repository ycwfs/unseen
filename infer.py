from ultralytics import YOLO
from tqdm import tqdm

# Load a pretrained YOLO11n model
model = YOLO("/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/yolol_rgb_ir_1280_ch6_splitval_mask_fuse_hsvaug/train_720_12802/weights/best.pt")
# Print the model summary
model.info(detailed=True, verbose=True, imgsz=[736, 1280])
# breakpoint()
# Run inference on 'bus.jpg' with arguments
res = model.predict("/data1/wangqiurui/code/ossutil-v1.7.19-linux-amd64/rgb_unseen/val/images/", 
                    save=True, imgsz=[736,1280], conf=0.25, stream=True, device="cuda:0", line_width=1, 
                    save_txt=True, save_conf=True, project='yolol_rgb_ir_1280_ch6_splitval_mask_fuse_hsvaug', name='train_720_1280_res')
for r in res:
    print(r.summary())
from ultralytics import YOLO
from tqdm import tqdm

# Load a pretrained YOLO11n model
model = YOLO("/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/yolos_rgb_1280/distill_ls_1280_all_halffl_4heads/weights/best.pt")
# Print the model summary
model.info(detailed=True, verbose=True, imgsz=[736, 1280])
# breakpoint()
# Run inference on 'bus.jpg' with arguments
res = model.predict("/data1/wangqiurui/code/ossutil-v1.7.19-linux-amd64/rgb_unseen/val/images", 
                    save=True, imgsz=[736,1280], conf=0.1, stream=True, device="cuda:0", line_width=1, 
                    save_txt=True, save_conf=True, project='/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/yolos_rgb_1280/distill_ls_1280_all_halffl_4heads', name='res_86')
for r in res:
    print(r.summary())
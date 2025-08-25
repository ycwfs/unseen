from ultralytics import YOLO
from tqdm import tqdm

# Load a pretrained YOLO11n model
model = YOLO("/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/yolos_rgb_1280/distill_ls_1280_all_halffl_4heads/weights/best.pt")
# Print the model summary
model.info(detailed=True, verbose=True, imgsz=[736, 1280])
# breakpoint()
# [720, 1280] 2 x [360, 640] 4 x [180, 320] 8 x [90, 160] 16 x [45, 80] 32 x [22, 40] 64 x [11, 20] 128 x [5, 10] 256 x [2, 4]
# Run inference on 'bus.jpg' with arguments
res = model.predict("/data1/wangqiurui/code/ossutil-v1.7.19-linux-amd64/rgb_unseen/val2/vis",
                    save=True, imgsz=[736,1280], conf=0.1, stream=True, device="cuda:0", line_width=1,
                    save_txt=True, save_conf=True, project='/data1/wangqiurui/code/competition/tianchi/unseen/yolov12/yolos_rgb_1280/', name='val2_res_938ttt')
for r in res:
    print(r.summary())
# Install
```bash
conda env create -f environment.yml
```
# Data Split
adjust data path in script first, then run
```bash
python dataset_splitter.py
```
# Config Adjust
adjust modal and data config first
- ./ultralytics/cfg/datasets/unseen.yaml
- ./ultralytics/cfg/models/v12/yolov12s.yaml
# Finetune YOLOV12
adjust modal and data path in script first
```bash
python ./scripts/train.py
```
# Compress
adjust modal and data path in script first
```bash
python compress.py
```

# Distillation
modify teacher and student path, distill config in script first
```bash
python distill.py
```

# Infer
modify ouput pt path to infer
```bash
python infer.py
```

# Process
modify pred_dir, and get final submit csv
```bash
cd result
python process_results.py
```


<!-- # Development
- fit infrared image fusion（conv + add）
  - through dataloader（check if ir path exist）
  - don't augment when use infrared image
  - don't use rect mode （config in build yolo dataset）
  - check amp fitting in predownload path
  - need further adjust in get_flops
- fit infrared image fusion (concat rgb 3 + ir 1 = 4, conv 4->32)
- fit infrared image fusion (concat rgb 3->3 + ir 3->1 = 4, conv 4->32)
- fit infrared image fusion (concat rgb 3->3 + ir 3->3 = 6, 6->activate which channel(fc), conv 6->32)
- data augment （copy small object region to blank space）may not good
- val data split（according to data distribution）
- plot infrared images batch（in plot_training_samples function）
- fit infrared image infer (modify LoadImagesAndVideos function and pre,postprocess)
- remove amp check
- train at (736,1280) latterbox size
- fit Albumentations, HorizontalFlip, RandomHSV augmeantations
- concat gray region into the ir region to fill the black region
- directly fuse rgb and ir and finrtune(base on rgb)
- fix transforms bug
- fix evaluation bug
- try maefuse rgb and ir(shit)

# Finial decision
- only use rgb image
- yolos train(1280,1280) infer(736,1280) 
- prune 40G flops 9 Mb params -> to 19G flops
- distill teacher(yolol 0.8141) -> student(yolov12s_rgb 0.7571)

# dif
  ~~12686~~
# TODO
- ~~smooth the filling operation~~
- ~~augmentation (ir+gray or ir+rgb) picture~~
- use full dataset to prune the yolo12s_rgb model
- continue finetune student(yolov12s_rgb 0.7571)
- exp on distillation layers

# tips
- x = box + cls + dfl loss
- 0.5 < feature loss < 1.5x -->
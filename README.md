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
- ./ultralytics/cfg/datasets/unseen_split.yaml
- ./ultralytics/cfg/models/v12/yolov12sir.yaml
# Finetune YOLOV12
adjust modal and data path in script first
```bash
python ./scripts/train_ir.py
```
# Compress
adjust modal and data path in script first
```bash
python compress.py
```
# Distillation
TDB
# Development
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

# dif
  12686
# TODO
- smooth the filling operation
- augmentation (ir+gray or ir+rgb) picture
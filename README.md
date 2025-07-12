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
- fit infrared image fusion (concat rgb 3 + ir 1 = 4, conv 4->32) (TBD)
- data augment （copy small object region to blank space）
- val data split（according to data distribution）
- save infrared images batch（in plot_training_samples function）
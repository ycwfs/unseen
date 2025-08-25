# Install
```bash
conda env create -f environment.yml
conda activate yolo12
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
# Quick Start - License Plate Recognition Training

## 🚀 Fastest Way to Train

### Option 1: Train from Scratch (Using Repository Components)

**Using YAML model configuration** (recommended for understanding the repo):

```bash
# From repository root
yolo train model=yolo11n.yaml data="ultralytics/engine/License Plate Recognition.v11i.yolov11/data.yaml" epochs=100
```

Or using Python:
```python
from ultralytics import YOLO
model = YOLO("yolo11n.yaml")  # From scratch, no pretrained weights
model.train(data="ultralytics/engine/License Plate Recognition.v11i.yolov11/data.yaml", epochs=100)
```

### Option 2: Using Pretrained Weights (Transfer Learning)

```bash
yolo train model=yolo11n.pt data="ultralytics/engine/License Plate Recognition.v11i.yolov11/data.yaml" epochs=100
```

### Option 3: Using the Training Scripts

1. Navigate to the dataset directory:
```bash
cd "ultralytics/engine/License Plate Recognition.v11i.yolov11"
```

2. Run the training script:
```bash
# Standard training (uses YOLO interface)
python train_license_plate.py

# Advanced training (uses DetectionTrainer directly)
python train_with_detection_trainer.py
```

## 📍 Where to Find Your Trained Model

After training completes, find your `.pt` model files here:

```
runs/detect/train/weights/
├── best.pt  ← USE THIS ONE (best validation performance)
└── last.pt  ← Last epoch checkpoint
```

**Full path example:**
```
C:\Users\User\Desktop\shahmir\ultralytics\runs\detect\train\weights\best.pt
```

## 🎯 Using Your Trained Model

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Predict on an image
results = model.predict("path/to/image.jpg", save=True)
```

Or via CLI:
```bash
yolo predict model=runs/detect/train/weights/best.pt source="path/to/image.jpg"
```

## ⚙️ Quick Tips

- **Training from scratch**: Use `.yaml` files (e.g., `yolo11n.yaml`) - no pretrained weights
- **Transfer learning**: Use `.pt` files (e.g., `yolo11n.pt`) - uses pretrained COCO weights
- **Out of memory?** Reduce batch size: `batch=8` or `batch=4`
- **Want better accuracy?** Use larger model: `yolo11s`, `yolo11m`, or `yolo11l`
- **Want faster training?** Use smaller image size: `imgsz=416`

## 📖 Full Documentation

- **`TRAINING_GUIDE.md`** - Detailed training instructions and examples
- **`TRAINING_WITH_REPO_COMPONENTS.md`** - Complete guide to using repository training infrastructure


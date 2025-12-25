# Training with Ultralytics Repository Components

This guide explains how to use the Ultralytics repository's complete training infrastructure to train a model from scratch on your custom dataset.

## 📚 Repository Training Architecture

The Ultralytics repository provides a complete training pipeline with the following components:

### 1. **Model Architectures** (`ultralytics/cfg/models/`)
Model architectures are defined in YAML configuration files. You can:
- Load from `.yaml` to create architecture from scratch (no pretrained weights)
- Load from `.pt` to use pretrained weights (transfer learning)

### 2. **Dataset Handling** (`ultralytics/data/`)
- `dataset.py`: YOLODataset class for loading images and labels
- `build.py`: Functions to build datasets and dataloaders
- `augment.py`: Data augmentation pipelines
- `utils.py`: Dataset utilities and format conversions

### 3. **Training Engine** (`ultralytics/engine/`)
- `trainer.py`: BaseTrainer class - core training loop, optimization, checkpointing
- `model.py`: Model class - high-level interface for training/validation/prediction
- `validator.py`: Validation logic

### 4. **Model-Specific Trainers** (`ultralytics/models/yolo/detect/train.py`)
- `DetectionTrainer`: Specialized trainer for object detection tasks
- Handles detection-specific dataset building, loss computation, validation

### 5. **Neural Network Modules** (`ultralytics/nn/`)
- `tasks.py`: Model architecture builders (DetectionModel, SegmentationModel, etc.)
- Various module definitions for building YOLO architectures

## 🚀 Training Methods

### Method 1: Train from Scratch (No Pretrained Weights)

**Using YAML model configuration:**

```python
from ultralytics import YOLO

# Load model architecture from YAML (no pretrained weights)
model = YOLO("yolo11n.yaml")  # Creates architecture from scratch

# Train on your dataset
results = model.train(
    data="ultralytics/engine/License Plate Recognition.v11i.yolov11/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
)
```

**Using CLI:**
```bash
yolo train model=yolo11n.yaml data="ultralytics/engine/License Plate Recognition.v11i.yolov11/data.yaml" epochs=100
```

### Method 2: Using the Training Infrastructure Directly

**Using DetectionTrainer directly:**

```python
from ultralytics.models.yolo.detect import DetectionTrainer

# Configure training
args = {
    "model": "yolo11n.yaml",  # or "yolo11n.pt" for pretrained
    "data": "ultralytics/engine/License Plate Recognition.v11i.yolov11/data.yaml",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "device": 0,  # GPU device
}

# Create trainer and start training
trainer = DetectionTrainer(overrides=args)
trainer.train()
```

### Method 3: Using BaseTrainer (Advanced)

**Low-level access to training infrastructure:**

```python
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.cfg import get_cfg

# Get default configuration
cfg = get_cfg()

# Override with your settings
cfg.update({
    "model": "yolo11n.yaml",
    "data": "ultralytics/engine/License Plate Recognition.v11i.yolov11/data.yaml",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
})

# Note: BaseTrainer is abstract, use DetectionTrainer instead
# This is shown for understanding the architecture
```

## 🔧 Available Model Architectures

You can train any of these architectures from scratch:

### YOLO11 Models
- `ultralytics/cfg/models/11/yolo11.yaml` - Base YOLO11 (scales: n, s, m, l, x)
- `ultralytics/cfg/models/11/yolo11-seg.yaml` - Segmentation
- `ultralytics/cfg/models/11/yolo11-pose.yaml` - Pose estimation
- `ultralytics/cfg/models/11/yolo11-obb.yaml` - Oriented bounding boxes
- `ultralytics/cfg/models/11/yolo11-cls.yaml` - Classification

### YOLO12 Models
- `ultralytics/cfg/models/12/yolo12.yaml` and variants

### Other Architectures
- YOLOv10, YOLOv9, YOLOv8, YOLOv5, etc. in their respective folders

**To use a specific scale (n/s/m/l/x), the system automatically resolves:**
- `yolo11n.yaml` → loads `yolo11.yaml` with scale='n'
- `yolo11s.yaml` → loads `yolo11.yaml` with scale='s'
- etc.

## 📂 Training Pipeline Flow

When you call `model.train()`, here's what happens:

1. **Configuration Loading** (`ultralytics/cfg/__init__.py`)
   - Loads default config from `default.yaml`
   - Merges with your provided arguments
   - Resolves model and dataset paths

2. **Model Creation** (`ultralytics/nn/tasks.py`)
   - If `.yaml`: Creates DetectionModel from scratch
   - If `.pt`: Loads pretrained weights into architecture
   - Sets number of classes from dataset config

3. **Dataset Building** (`ultralytics/data/build.py`)
   - `build_yolo_dataset()`: Creates YOLODataset instances
   - `build_dataloader()`: Creates PyTorch DataLoaders
   - Handles data augmentation, caching, batching

4. **Training Loop** (`ultralytics/engine/trainer.py`)
   - `BaseTrainer.train()`: Main training loop
   - Forward pass, loss computation, backpropagation
   - Validation, checkpointing, logging

5. **Model Saving**
   - Saves to `runs/detect/train/weights/`
   - `best.pt`: Best model based on validation metrics
   - `last.pt`: Last epoch checkpoint

## 🎯 Complete Training Example

Here's a complete example using repository components:

```python
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

# Path to your dataset config
DATA_YAML = Path("ultralytics/engine/License Plate Recognition.v11i.yolov11/data.yaml")

# Option 1: Using YOLO high-level interface (recommended)
model = YOLO("yolo11n.yaml")  # Architecture from scratch, no pretrained weights
results = model.train(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project="runs/detect",
    name="license_plate_training",
)

print(f"Training completed! Best model: {results.save_dir}/weights/best.pt")

# Option 2: Using DetectionTrainer directly (more control)
trainer = DetectionTrainer(overrides={
    "model": "yolo11n.yaml",  # or "ultralytics/cfg/models/11/yolo11.yaml" with scale='n'
    "data": str(DATA_YAML),
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "device": 0,
    "project": "runs/detect",
    "name": "license_plate_custom",
})
trainer.train()
```

## 🔍 Understanding Model Configuration

Model YAML files define architecture:

```yaml
# yolo11.yaml structure
nc: 80  # number of classes (overridden by your data.yaml)
scales:
  n: [0.50, 0.25, 1024]  # [depth, width, max_channels]
  s: [0.50, 0.50, 1024]
  # ...

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]
  # ...

head:
  # Detection head layers
  - [[16, 19, 22], 1, Detect, [nc]]
```

When you use `yolo11n.yaml`, the system:
1. Loads `yolo11.yaml`
2. Applies scale='n' (depth=0.5, width=0.25)
3. Creates the model architecture
4. Initializes weights randomly (no pretraining)

## 📊 Dataset Configuration

Your `data.yaml` should follow this structure:

```yaml
# Paths relative to data.yaml location
train: train/images
val: valid/images
test: test/images  # optional

# Number of classes
nc: 1

# Class names
names:
  0: License_Plate
```

The repository's dataset loader (`YOLODataset`) automatically:
- Finds images in specified directories
- Loads corresponding label files from `labels/` directories
- Applies augmentations during training
- Handles caching for faster training

## 🛠️ Customizing Training

### Custom Augmentations

```python
from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
results = model.train(
    data="your_data.yaml",
    epochs=100,
    hsv_h=0.015,      # Hue augmentation
    hsv_s=0.7,        # Saturation augmentation
    hsv_v=0.4,        # Value augmentation
    degrees=10.0,     # Rotation augmentation
    translate=0.1,    # Translation augmentation
    scale=0.5,        # Scaling augmentation
    fliplr=0.5,       # Horizontal flip probability
    mosaic=1.0,       # Mosaic augmentation probability
)
```

### Training Configuration Options

All training parameters are in `ultralytics/cfg/default.yaml`. Key options:

- **Model**: `model` - Path to .yaml or .pt file
- **Data**: `data` - Path to dataset YAML
- **Training**: `epochs`, `batch`, `imgsz`, `workers`
- **Optimization**: `optimizer`, `lr0`, `momentum`, `weight_decay`
- **Augmentation**: `hsv_h`, `hsv_s`, `degrees`, `translate`, etc.
- **Device**: `device` - GPU/CPU selection
- **Logging**: `project`, `name`, `exist_ok`

## 📍 Model Output Locations

After training, find your models:

```
runs/detect/train/weights/
├── best.pt       # Best model (use this for inference)
└── last.pt       # Last epoch checkpoint

runs/detect/train/
├── args.yaml     # Training configuration used
├── results.csv   # Training metrics per epoch
├── confusion_matrix.png
├── results.png   # Training curves
└── ...
```

## 🔗 Key Repository Files Reference

- **Model Loading**: `ultralytics/engine/model.py` - `Model._new()`, `Model._load()`
- **Model Architecture**: `ultralytics/nn/tasks.py` - `DetectionModel.__init__()`
- **Training Loop**: `ultralytics/engine/trainer.py` - `BaseTrainer.train()`
- **Dataset Building**: `ultralytics/data/build.py` - `build_yolo_dataset()`
- **Data Loading**: `ultralytics/data/dataset.py` - `YOLODataset`
- **Detection Trainer**: `ultralytics/models/yolo/detect/train.py` - `DetectionTrainer`

## 💡 Tips

1. **From Scratch vs Pretrained**: 
   - `.yaml` = train from scratch (random initialization)
   - `.pt` = transfer learning (pretrained weights)

2. **Model Sizes**: Start with `yolo11n.yaml` (nano) for faster training, scale up if needed

3. **Batch Size**: Adjust based on GPU memory. Reduce if you get OOM errors

4. **Epochs**: Monitor validation metrics. Use early stopping (`patience` parameter)

5. **Debugging**: Set `verbose=True` and check `runs/detect/train/results.csv`

## 🎓 Learning the Codebase

To understand how training works internally:

1. Start with `YOLO.train()` in `ultralytics/engine/model.py` (line ~711)
2. Follow to `DetectionTrainer` in `ultralytics/models/yolo/detect/train.py`
3. See `BaseTrainer.train()` in `ultralytics/engine/trainer.py` (line ~200+)
4. Check dataset building in `ultralytics/data/build.py`
5. Explore model architecture in `ultralytics/nn/tasks.py`

This gives you a complete understanding of the training pipeline!


# License Plate Recognition - Training Guide

This directory contains everything you need to train a YOLO model on your License Plate Recognition dataset using the Ultralytics repository.

## 📁 Files in This Directory

- **`data.yaml`** - Dataset configuration file (paths to train/val/test sets and class names)
- **`train_license_plate.py`** - Main training script (recommended for beginners)
- **`train_with_detection_trainer.py`** - Advanced training script using DetectionTrainer directly
- **`QUICK_START.md`** - Quick reference guide
- **`TRAINING_GUIDE.md`** - Detailed training instructions
- **`TRAINING_WITH_REPO_COMPONENTS.md`** - Complete guide to using repository training infrastructure

## 🎯 Quick Start

### Train from Scratch (No Pretrained Weights)

```bash
# From repository root directory
yolo train model=yolo11n.yaml data="ultralytics/engine/License Plate Recognition.v11i.yolov11/data.yaml" epochs=100
```

Or using Python:
```python
from ultralytics import YOLO
model = YOLO("yolo11n.yaml")  # Creates architecture from scratch
model.train(data="ultralytics/engine/License Plate Recognition.v11i.yolov11/data.yaml", epochs=100)
```

### Using the Training Script

```bash
cd "ultralytics/engine/License Plate Recognition.v11i.yolov11"
python train_license_plate.py
```

## 📍 Where to Find Your Trained Model

After training, your model files will be saved in:

```
runs/detect/train/weights/
├── best.pt  ← Use this for inference (best validation performance)
└── last.pt  ← Last epoch checkpoint
```

**Full path example:**
```
C:\Users\User\Desktop\shahmir\ultralytics\runs\detect\train\weights\best.pt
```

## 🔑 Key Concepts

### Training from Scratch vs Transfer Learning

1. **Training from Scratch** (`.yaml` files)
   - Creates model architecture from configuration
   - Random weight initialization
   - Example: `yolo11n.yaml`
   - Uses repository's model building components directly

2. **Transfer Learning** (`.pt` files)
   - Loads pretrained weights from COCO dataset
   - Fine-tunes on your dataset
   - Example: `yolo11n.pt`
   - Faster convergence, but uses pretrained weights

### Repository Training Components

The Ultralytics repository provides a complete training infrastructure:

1. **Model Architectures** (`ultralytics/cfg/models/`)
   - YAML files defining model architectures
   - Scales: n (nano), s (small), m (medium), l (large), x (extra large)

2. **Training Engine** (`ultralytics/engine/`)
   - `trainer.py`: BaseTrainer - core training loop
   - `model.py`: Model class - high-level training interface
   - `validator.py`: Validation logic

3. **Detection Trainer** (`ultralytics/models/yolo/detect/train.py`)
   - DetectionTrainer - specialized for object detection
   - Handles dataset building, loss computation, validation

4. **Dataset Handling** (`ultralytics/data/`)
   - `dataset.py`: YOLODataset class
   - `build.py`: Dataset and dataloader builders
   - `augment.py`: Data augmentation pipelines

5. **Model Building** (`ultralytics/nn/`)
   - `tasks.py`: DetectionModel, SegmentationModel, etc.
   - Architecture builders that create models from YAML configs

## 📚 Documentation

- **Quick Start**: See `QUICK_START.md` for fastest way to train
- **Training Guide**: See `TRAINING_GUIDE.md` for detailed instructions
- **Repository Components**: See `TRAINING_WITH_REPO_COMPONENTS.md` for understanding how the repository works internally

## 🛠️ Configuration

### Dataset Configuration (`data.yaml`)

```yaml
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  0: License_Plate
```

### Model Selection

Choose model size based on your needs:
- **`yolo11n.yaml`** - Nano (fastest, smallest, least accurate)
- **`yolo11s.yaml`** - Small (balanced)
- **`yolo11m.yaml`** - Medium (better accuracy)
- **`yolo11l.yaml`** - Large (high accuracy)
- **`yolo11x.yaml`** - Extra Large (best accuracy, slowest)

## 💡 Tips

1. Start with `yolo11n.yaml` for faster training and understanding
2. Monitor training metrics in `runs/detect/train/results.csv`
3. Adjust batch size if you get out-of-memory errors
4. Use early stopping with `patience` parameter
5. Check `runs/detect/train/` for training plots and metrics

## 🔍 Understanding the Training Pipeline

When you run training, here's what happens:

1. **Model Loading**: Architecture is built from YAML config
2. **Dataset Building**: Images and labels are loaded and organized
3. **Data Loading**: PyTorch DataLoaders are created with augmentations
4. **Training Loop**: Forward pass, loss computation, backpropagation
5. **Validation**: Model is evaluated on validation set each epoch
6. **Checkpointing**: Best and last models are saved

For detailed understanding, see `TRAINING_WITH_REPO_COMPONENTS.md`.

## 🎓 Next Steps

1. Run training using one of the methods above
2. Monitor training progress in the console or `results.csv`
3. Use `best.pt` for inference after training completes
4. Experiment with different model sizes and hyperparameters

For questions or issues, refer to the Ultralytics documentation: https://docs.ultralytics.com/


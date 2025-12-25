# License Plate Recognition - Training Guide

This guide explains how to train a YOLO model on your custom License Plate Recognition dataset.

## Dataset Structure

Your dataset is located at:
```
ultralytics/engine/License Plate Recognition.v11i.yolov11/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Training Methods

### Method 1: Using Command Line Interface (CLI) - Recommended

**From the repository root directory:**

```bash
# Basic training with default settings
yolo train model=yolo11n.pt data="ultralytics/engine/License Plate Recognition.v11i.yolov11/data.yaml" epochs=100 imgsz=640

# Training with more options
yolo train model=yolo11n.pt data="ultralytics/engine/License Plate Recognition.v11i.yolov11/data.yaml" epochs=100 imgsz=640 batch=16 patience=50

# Use a larger model for better accuracy (slower)
yolo train model=yolo11s.pt data="ultralytics/engine/License Plate Recognition.v11i.yolov11/data.yaml" epochs=100 imgsz=640
```

**Or from the dataset directory:**

```bash
cd "ultralytics/engine/License Plate Recognition.v11i.yolov11"
yolo train model=yolo11n.pt data=data.yaml epochs=100 imgsz=640
```

### Method 2: Using Python Script

Create a Python script (e.g., `train_license_plate.py`):

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11n.pt")  # or yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt for larger models

# Train the model
results = model.train(
    data="ultralytics/engine/License Plate Recognition.v11i.yolov11/data.yaml",  # dataset config file
    epochs=100,          # number of training epochs
    imgsz=640,          # image size for training
    batch=16,           # batch size (adjust based on your GPU memory)
    patience=50,        # early stopping patience
    save=True,          # save checkpoints
    project="runs/detect",  # project directory
    name="license_plate",   # experiment name
    exist_ok=True,      # overwrite existing experiment
)

print("Training completed!")
print(f"Best model saved at: {results.save_dir}/weights/best.pt")
```

Run it with:
```bash
python train_license_plate.py
```

## Where to Find Your Trained Model (.pt file)

After training completes, your trained model files will be saved in:

```
runs/detect/train/weights/
├── best.pt      # Best model based on validation metrics (USE THIS ONE)
└── last.pt      # Last epoch checkpoint
```

Or if you specified a custom name:
```
runs/detect/license_plate/weights/
├── best.pt
└── last.pt
```

**The `best.pt` file is the one you should use** - it has the best performance on your validation set.

## Model Sizes (Choose Based on Your Needs)

- `yolo11n.pt` - Nano (smallest, fastest, least accurate)
- `yolo11s.pt` - Small (balanced)
- `yolo11m.pt` - Medium (better accuracy)
- `yolo11l.pt` - Large (high accuracy)
- `yolo11x.pt` - Extra Large (best accuracy, slowest)

## Training Parameters Explained

- `epochs`: Number of training iterations over the entire dataset (default: 100)
- `imgsz`: Image size for training (default: 640)
- `batch`: Batch size - increase if you have more GPU memory (default: 16)
- `patience`: Early stopping patience - stop if no improvement for N epochs (default: 50)
- `device`: Device to train on - `0` for GPU 0, `cpu` for CPU (auto-detected by default)

## Monitoring Training

During training, you can monitor progress:

1. **Console output**: Real-time training metrics printed to console
2. **TensorBoard**: If enabled, view at `runs/detect/train/`
3. **Results**: Saved in `runs/detect/train/results.csv`

## Using Your Trained Model

Once training is complete, use your model like this:

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run inference on an image
results = model.predict("path/to/image.jpg")

# Run inference on a video
results = model.predict("path/to/video.mp4", save=True)
```

Or via CLI:
```bash
yolo predict model=runs/detect/train/weights/best.pt source="path/to/image.jpg"
```

## Troubleshooting

1. **Out of memory error**: Reduce `batch` size (e.g., `batch=8` or `batch=4`)
2. **Slow training**: Use a smaller model (`yolo11n.pt`) or reduce `imgsz` (e.g., `imgsz=416`)
3. **Paths not found**: Make sure you're running from the correct directory or use absolute paths in `data.yaml`

## Additional Resources

- Full documentation: https://docs.ultralytics.com/modes/train/
- Ultralytics GitHub: https://github.com/ultralytics/ultralytics
- Community Forum: https://community.ultralytics.com/


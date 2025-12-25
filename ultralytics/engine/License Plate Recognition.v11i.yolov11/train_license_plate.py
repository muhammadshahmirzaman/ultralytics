"""
Training script for License Plate Recognition using YOLO11
This script trains a YOLO model from scratch using the Ultralytics repository components
on the custom License Plate Recognition dataset.

You can train from scratch (using .yaml) or use pretrained weights (using .pt)
"""

from pathlib import Path
from ultralytics import YOLO

# Get the current directory (dataset directory)
DATASET_DIR = Path(__file__).parent
DATA_YAML = DATASET_DIR / "data.yaml"

# ===== CONFIGURATION =====
# Option 1: Train from scratch (no pretrained weights) - RECOMMENDED for understanding the repo
MODEL_CONFIG = "yolo11n.yaml"  # Creates architecture from scratch

# Option 2: Use pretrained weights (transfer learning) - UNCOMMENT to use
# MODEL_CONFIG = "yolo11n.pt"  # Uses pretrained COCO weights

# Other model sizes: yolo11s.yaml, yolo11m.yaml, yolo11l.yaml, yolo11x.yaml
# Or with pretrained: yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16  # Adjust based on your GPU memory (reduce if out of memory)
PATIENCE = 50  # Early stopping patience

def main():
    """Train YOLO model on License Plate Recognition dataset using repository components."""
    
    print("=" * 70)
    print("License Plate Recognition - YOLO11 Training")
    print("Training from scratch using Ultralytics repository components")
    print("=" * 70)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Data config: {DATA_YAML}")
    print(f"Model: {MODEL_CONFIG}")
    if MODEL_CONFIG.endswith('.yaml'):
        print("Mode: Training from scratch (no pretrained weights)")
    else:
        print("Mode: Transfer learning (using pretrained weights)")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 70)
    
    # Load model architecture from YAML (from scratch) or pretrained .pt file
    print(f"\nLoading model: {MODEL_CONFIG}")
    model = YOLO(MODEL_CONFIG)
    
    # Train the model
    print(f"\nStarting training...")
    print(f"Training data: {DATA_YAML}")
    
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        save=True,
        project="runs/detect",
        name="license_plate",
        exist_ok=True,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # Print where the model was saved
    best_model_path = results.save_dir / "weights" / "best.pt"
    last_model_path = results.save_dir / "weights" / "last.pt"
    
    print(f"\nBest model saved at: {best_model_path}")
    print(f"Last checkpoint saved at: {last_model_path}")
    print(f"\nUse the best.pt file for inference!")
    print(f"Example: model = YOLO('{best_model_path}')")
    print("=" * 60)


if __name__ == "__main__":
    main()


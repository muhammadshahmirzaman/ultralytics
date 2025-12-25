"""
Advanced training script using DetectionTrainer directly
This demonstrates using the repository's training components at a lower level
for more control over the training process.
"""

from pathlib import Path
from ultralytics.models.yolo.detect import DetectionTrainer

# Get the current directory (dataset directory)
DATASET_DIR = Path(__file__).parent
DATA_YAML = DATASET_DIR / "data.yaml"

def main():
    """Train using DetectionTrainer for direct access to training components."""
    
    print("=" * 70)
    print("License Plate Recognition - Using DetectionTrainer")
    print("Direct access to Ultralytics training infrastructure")
    print("=" * 70)
    
    # Configure training using DetectionTrainer
    # This gives you direct access to the training components
    training_config = {
        # Model configuration - use .yaml for from scratch, .pt for pretrained
        "model": "yolo11n.yaml",  # Train from scratch
        # "model": "yolo11n.pt",  # Use pretrained weights (uncomment to use)
        
        # Dataset configuration
        "data": str(DATA_YAML),
        
        # Training hyperparameters
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "patience": 50,  # Early stopping
        
        # Device
        "device": 0,  # GPU 0, or 'cpu' for CPU training
        
        # Output directories
        "project": "runs/detect",
        "name": "license_plate_advanced",
        "exist_ok": True,
        
        # Optimization
        "optimizer": "auto",  # auto, SGD, Adam, AdamW, etc.
        "lr0": 0.01,  # Initial learning rate
        "momentum": 0.937,
        "weight_decay": 0.0005,
        
        # Data augmentation
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        
        # Other settings
        "workers": 8,  # Data loading workers
        "verbose": True,
    }
    
    print(f"\nTraining Configuration:")
    print(f"  Model: {training_config['model']}")
    print(f"  Data: {training_config['data']}")
    print(f"  Epochs: {training_config['epochs']}")
    print(f"  Batch size: {training_config['batch']}")
    print(f"  Image size: {training_config['imgsz']}")
    print(f"  Device: {training_config['device']}")
    print("=" * 70)
    
    # Create DetectionTrainer instance
    # This internally uses BaseTrainer and all the repository components:
    # - Model loading: DetectionModel from ultralytics/nn/tasks.py
    # - Dataset building: build_yolo_dataset from ultralytics/data/build.py
    # - Training loop: BaseTrainer.train() from ultralytics/engine/trainer.py
    print("\nCreating DetectionTrainer...")
    trainer = DetectionTrainer(overrides=training_config)
    
    # Start training
    # This runs the complete training pipeline:
    # 1. Loads model architecture from YAML
    # 2. Builds datasets (train/val)
    # 3. Creates data loaders
    # 4. Sets up optimizer and scheduler
    # 5. Runs training loop with validation
    # 6. Saves checkpoints (best.pt, last.pt)
    print("\nStarting training...")
    trainer.train()
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    
    # Access training results
    best_model_path = trainer.best  # Path to best.pt
    save_dir = trainer.save_dir     # Directory with all results
    
    print(f"\nResults saved to: {save_dir}")
    print(f"Best model: {best_model_path}")
    print(f"Last checkpoint: {trainer.last}")
    print("\nYou can access trainer attributes:")
    print(f"  - trainer.model: The trained model")
    print(f"  - trainer.data: Dataset configuration")
    print(f"  - trainer.metrics: Training metrics")
    print("=" * 70)


if __name__ == "__main__":
    main()


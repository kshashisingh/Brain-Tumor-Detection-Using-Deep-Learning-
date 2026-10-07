#!/usr/bin/env python3
"""
BRAIN TUMOR CLASSIFICATION - COMPLETE RESEARCH PIPELINE
FIXED VERSION - Compatible with RTX 4070 SUPER (12GB) and Windows
"""

# ============================================================================
# USER CONFIGURATION - FIXED FOR 12GB GPU AND WINDOWS
# ============================================================================

# INPUT PATHS
TRAIN_DATA_PATH = r"C:\Users\nielitpatna\Desktop\new brain\Training"
TEST_DATA_PATH = r"C:\Users\nielitpatna\Desktop\new brain\Testing"

# OUTPUT PATH (Will be created, handles existing file/folder automatically)
OUTPUT_BASE_PATH = r"C:\Users\nielitpatna\Desktop\new brain\main_brain_output"

# MODELS TO TRAIN - Reduced list for 12GB GPU stability
# Removed: b4, b5, b6, b7, resnet152, densenet201, vit_l_16, vit_l_32 (too large)
MODELS_TO_TRAIN = [
    # EfficientNet Family (b0-b3 safe for 12GB)
    'efficientnet_b0',      # Fast, efficient
    'efficientnet_b3',      # Higher accuracy
    
    # ResNet Family
    'resnet50',             # Balanced
    'resnet101',            # Higher capacity
    
    # DenseNet Family
    'densenet121',          # Memory efficient
    'densenet161',          # Wider
    
    # MobileNet Family (Lightweight)
    'mobilenet_v2',         # Fast inference
    'mobilenet_v3_large',   # Better accuracy
    
    # VGG Family
    'vgg16',                # Classic architecture
    
    # Vision Transformers (Base only for 12GB)
    'vit_b_16',             # Base ViT
    'vit_b_32',             # Base ViT 32-patch (more efficient)
]

# TRAINING CONFIGURATION - FIXED FOR 12GB GPU
BATCH_SIZE = 16                  # Reduced from 64 (memory constraint)
IMAGE_SIZE = 224                 # Reduced from 299 (standard size, less memory)
VALIDATION_SPLIT = 0.2           # 20% validation
RANDOM_SEED = 42                 # Reproducibility

# TRAINING PHASES
EPOCHS_PHASE1 = 10               # Reduced for faster iteration
EPOCHS_PHASE2 = 20               # Reduced but sufficient for fine-tuning

# HYPERPARAMETERS
LEARNING_RATE_PHASE1 = 1e-3      # Standard for smaller batches
LEARNING_RATE_PHASE2 = 1e-5      # Finer tuning
WEIGHT_DECAY = 1e-4              # L2 regularization
LABEL_SMOOTHING = 0.1            # Prevents overconfidence
DROPOUT_RATE = 0.5               # Increased regularization
GRADIENT_CLIP = 1.0              # Prevents exploding gradients
EARLY_STOPPING_PATIENCE = 7      # Reduced patience
GRADIENT_ACCUMULATION_STEPS = 1  # Can increase to simulate larger batches

# WINDOWS + 12GB SPECIFIC SETTINGS - CRITICAL FIXES
NUM_WORKERS = 0                  # FIXED: Must be 0 on Windows to avoid shared memory errors
PIN_MEMORY = False               # FIXED: Disable on Windows with num_workers=0
PERSISTENT_WORKERS = False       # FIXED: Only works with num_workers > 0
MIXED_PRECISION = True           # FP16 training for 2x speed and half memory
PREFETCH_FACTOR = 2              # Not used when num_workers=0

# MEMORY MANAGEMENT
EMPTY_CACHE_FREQUENCY = 3        # Clear cache every N epochs
MAX_BATCH_SIZE_FOR_LARGE_MODELS = 8  # Auto-reduce for large models

# FEATURE TOGGLES
ENABLE_THRESHOLD_TUNING = True       # Optimize decision thresholds per class
ENABLE_EMBEDDING_EXTRACTION = True   # t-SNE/UMAP visualizations
ENABLE_ENSEMBLE = True               # Combine all models
ENABLE_DETAILED_LOGGING = True       # Save detailed metrics
ENABLE_TTA = True                    # Test Time Augmentation

# ============================================================================
# IMPORTS & SETUP
# ============================================================================

import os
import sys
import shutil
import random
import numpy as np
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict, Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR, StepLR
import torchvision
from torchvision import datasets, transforms, models
from torchvision.models import (VGG16_BN_Weights, VGG19_BN_Weights,
                               ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, 
                               ResNet101_Weights, ResNet152_Weights,
                               DenseNet121_Weights, DenseNet161_Weights, 
                               DenseNet169_Weights, DenseNet201_Weights,
                               MobileNet_V2_Weights, MobileNet_V3_Small_Weights, 
                               MobileNet_V3_Large_Weights,
                               EfficientNet_B0_Weights, EfficientNet_B1_Weights,
                               EfficientNet_B2_Weights, EfficientNet_B3_Weights,
                               EfficientNet_B4_Weights, EfficientNet_B5_Weights,
                               EfficientNet_B6_Weights, EfficientNet_B7_Weights,
                               ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights,
                               ViT_L_32_Weights)

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           confusion_matrix, classification_report, roc_curve, 
                           auc, precision_recall_curve, average_precision_score,
                           cohen_kappa_score, matthews_corrcoef, log_loss,
                           f1_score)
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
import pandas as pd

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("BRAIN TUMOR CLASSIFICATION - 12GB GPU OPTIMIZED RESEARCH PIPELINE")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"CUDA Memory: {total_mem:.2f} GB")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    
    # Memory optimizations for 12GB
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set memory fraction to leave headroom (critical for 12GB)
    torch.cuda.set_per_process_memory_fraction(0.85, 0)
    
    # Enable expandable segments to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    if MIXED_PRECISION:
        print("Mixed Precision (FP16): Enabled")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE} | Image Size: {IMAGE_SIZE} | Workers: {NUM_WORKERS}")
print("="*80)

# ============================================================================
# ROBUST DIRECTORY SETUP (WINDOWS FIX)
# ============================================================================

def robust_makedirs(path: str, exist_ok: bool = True) -> None:
    """
    Robust directory creation that handles Windows path issues.
    Handles cases where path exists as file, or parent doesn't exist.
    """
    try:
        if os.path.exists(path) and os.path.isfile(path):
            print(f"[WARNING] Path exists as file, removing: {path}")
            os.remove(path)
        os.makedirs(path, exist_ok=exist_ok)
    except PermissionError as e:
        print(f"[ERROR] Permission denied for {path}: {e}")
        import tempfile
        fallback = os.path.join(tempfile.gettempdir(), "BrainTumor_Results")
        print(f"[FALLBACK] Using: {fallback}")
        os.makedirs(fallback, exist_ok=True)
        return fallback
    except Exception as e:
        print(f"[ERROR] Cannot create {path}: {e}")
        raise

def setup_output_directories(base_path: str) -> Dict[str, str]:
    """
    Create all necessary output directories with error handling.
    """
    print(f"\n[SETUP] Creating output directories...")
    
    if os.path.exists(base_path):
        if os.path.isfile(base_path):
            print(f"[INFO] Removing existing file at output path")
            os.remove(base_path)
        else:
            print(f"[INFO] Output directory already exists, will reuse")
    
    robust_makedirs(base_path)
    
    dirs = {
        'root': base_path,
        'models': os.path.join(base_path, '01_saved_models'),
        'plots_training': os.path.join(base_path, '02_training_plots'),
        'plots_evaluation': os.path.join(base_path, '03_evaluation_plots'),
        'plots_thresholds': os.path.join(base_path, '04_threshold_analysis'),
        'embeddings': os.path.join(base_path, '05_embeddings'),
        'ensemble': os.path.join(base_path, '06_ensemble_results'),
        'logs': os.path.join(base_path, '07_logs'),
        'reports': os.path.join(base_path, '08_reports'),
        'checkpoints': os.path.join(base_path, '09_checkpoints')
    }
    
    for name, path in dirs.items():
        robust_makedirs(path)
        print(f"  ✓ {name:20s} -> {path}")
    
    return dirs

OUTPUT_DIRS = setup_output_directories(OUTPUT_BASE_PATH)

def set_global_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"\n[SEED] Random seed set to {seed}")

set_global_seed(RANDOM_SEED)

# ============================================================================
# DATA PIPELINE WITH AUGMENTATION - FIXED FOR WINDOWS
# ============================================================================

class MedicalImageTransforms:
    """
    Medical-grade image transformations optimized for MRI scans.
    Conservative augmentations to preserve diagnostic features.
    """
    
    @staticmethod
    def get_training_transforms(image_size: int = 224) -> transforms.Compose:
        """
        Training transforms with medical-safe augmentations.
        """
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            
            # Geometric augmentations (preserve structural integrity)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=(-5, 5)
            ),
            
            # Color augmentations (careful with medical images)
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            
            # Additional augmentations
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.2),
            
            transforms.ToTensor(),  # FIXED: Added missing ToTensor
            
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
        ])
    
    @staticmethod
    def get_validation_transforms(image_size: int = 224) -> transforms.Compose:
        """Validation/Test transforms - no augmentation."""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def get_test_time_augmentation_transforms(image_size: int = 224) -> List[transforms.Compose]:
        """
        Multiple transforms for test-time augmentation (TTA).
        Returns list of transforms to apply during inference.
        """
        base_transforms = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        
        tta_transforms = [
            transforms.Compose(base_transforms),
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=(-10, 10)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ]
        return tta_transforms

def analyze_dataset_structure(path: str) -> Dict[str, Any]:
    """
    Analyze dataset directory structure and provide statistics.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path not found: {path}")
    
    try:
        dataset = datasets.ImageFolder(path)
        classes = dataset.classes
        class_to_idx = dataset.class_to_idx
        
        class_counts = Counter([s[1] for s in dataset.samples])
        class_distribution = {classes[i]: count for i, count in class_counts.items()}
        
        return {
            'total_samples': len(dataset),
            'num_classes': len(classes),
            'classes': classes,
            'class_to_idx': class_to_idx,
            'class_distribution': class_distribution
        }
    except Exception as e:
        print(f"[ERROR] Cannot analyze dataset at {path}: {e}")
        raise

def create_data_loaders(
    train_path: str,
    test_path: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    image_size: int = 224,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], int]:
    """
    Create train, validation, and test dataloaders with proper splits.
    FIXED: Windows-compatible with num_workers=0
    """
    print(f"\n{'='*80}")
    print("DATA LOADING - WINDOWS/12GB OPTIMIZED")
    print(f"{'='*80}")
    
    for path, name in [(train_path, 'Training'), (test_path, 'Testing')]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{name} directory not found: {path}")
        print(f"[OK] {name}: {path}")
    
    train_info = analyze_dataset_structure(train_path)
    test_info = analyze_dataset_structure(test_path)
    
    print(f"\n[DATASET INFO]")
    print(f"Classes: {train_info['classes']}")
    print(f"Training samples: {train_info['total_samples']}")
    print(f"Test samples: {test_info['total_samples']}")
    
    if set(train_info['classes']) != set(test_info['classes']):
        raise ValueError("Training and test classes do not match!")
    
    train_transform = MedicalImageTransforms.get_training_transforms(image_size)
    val_transform = MedicalImageTransforms.get_validation_transforms(image_size)
    
    full_train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    
    total_train = len(full_train_dataset)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    val_dataset.dataset.transform = val_transform
    
    test_dataset = datasets.ImageFolder(test_path, transform=val_transform)
    
    print(f"[SPLIT] Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")
    
    # Handle class imbalance with weighted sampling
    train_targets = [full_train_dataset.targets[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_targets)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[train_targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # FIXED: Windows-compatible DataLoaders (num_workers=0, pin_memory=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,  # FIXED: 0 for Windows
        pin_memory=PIN_MEMORY,    # FIXED: False for Windows
        persistent_workers=PERSISTENT_WORKERS,  # FIXED: False
        prefetch_factor=None,     # FIXED: None when num_workers=0
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY
    )
    
    class_names = train_info['classes']
    num_classes = len(class_names)
    
    return train_loader, val_loader, test_loader, class_names, num_classes

# ============================================================================
# COMPREHENSIVE MODEL FACTORY - ALL MODELS ENABLED FOR 12GB
# ============================================================================

class ArchitectureFactory:
    """
    Factory for creating various deep learning architectures 
    with custom classification heads.
    Supports CNNs (VGG, ResNet, DenseNet, MobileNet, EfficientNet) 
    and Transformers (ViT).
    """
    
    ARCHITECTURES = {
        'vgg': ['vgg16', 'vgg19'],
        'resnet': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
        'densenet': ['densenet121', 'densenet161', 'densenet169', 'densenet201'],
        'mobilenet': ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'],
        'efficientnet': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
                        'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                        'efficientnet_b6', 'efficientnet_b7'],
        'vit': ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']
    }
    
    @classmethod
    def create_model(cls, model_name: str, num_classes: int, dropout: float = 0.5) -> nn.Module:
        """
        Create a model with custom classification head.
        """
        print(f"\n[MODEL BUILDER] Creating {model_name} for 12GB GPU...")
        
        if model_name not in [m for sublist in cls.ARCHITECTURES.values() for m in sublist]:
            raise ValueError(f"Model {model_name} not supported. Choose from: {cls.ARCHITECTURES}")
        
        model = None
        
        try:
            # VGG Family
            if model_name == 'vgg16':
                model = models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
                num_features = model.classifier[6].in_features
                model.classifier[6] = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'vgg19':
                model = models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
                num_features = model.classifier[6].in_features
                model.classifier[6] = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
            
            # ResNet Family
            elif model_name == 'resnet18':
                model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
                num_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'resnet34':
                model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
                num_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'resnet50':
                model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                num_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'resnet101':
                model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
                num_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'resnet152':
                model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
                num_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
            
            # DenseNet Family
            elif model_name == 'densenet121':
                model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
                num_features = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'densenet161':
                model = models.densenet161(weights=DenseNet161_Weights.DEFAULT)
                num_features = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'densenet169':
                model = models.densenet169(weights=DenseNet169_Weights.DEFAULT)
                num_features = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'densenet201':
                model = models.densenet201(weights=DenseNet201_Weights.DEFAULT)
                num_features = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
            
            # MobileNet Family
            elif model_name == 'mobilenet_v2':
                model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
                num_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'mobilenet_v3_small':
                model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
                num_features = model.classifier[3].in_features
                model.classifier[3] = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'mobilenet_v3_large':
                model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
                num_features = model.classifier[3].in_features
                model.classifier[3] = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
            
            # EfficientNet Family
            elif model_name == 'efficientnet_b0':
                model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
                num_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'efficientnet_b1':
                model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
                num_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'efficientnet_b2':
                model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
                num_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'efficientnet_b3':
                model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
                num_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'efficientnet_b4':
                model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
                num_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'efficientnet_b5':
                model = models.efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
                num_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'efficientnet_b6':
                model = models.efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
                num_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'efficientnet_b7':
                model = models.efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
                num_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
            
            # Vision Transformers
            elif model_name == 'vit_b_16':
                model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
                num_features = model.heads.head.in_features
                model.heads.head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'vit_b_32':
                model = models.vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
                num_features = model.heads.head.in_features
                model.heads.head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'vit_l_16':
                model = models.vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
                num_features = model.heads.head.in_features
                model.heads.head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
                
            elif model_name == 'vit_l_32':
                model = models.vit_l_32(weights=ViT_L_32_Weights.DEFAULT)
                num_features = model.heads.head.in_features
                model.heads.head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(num_features, num_classes)
                )
            
            if model is None:
                raise RuntimeError(f"Model {model_name} creation failed")
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size_mb = total_params * 4 / 1024 / 1024
            
            print(f"  ✓ Total parameters: {total_params:,}")
            print(f"  ✓ Trainable parameters: {trainable_params:,}")
            print(f"  ✓ Model size (est.): {model_size_mb:.2f} MB")
            
            return model
            
        except Exception as e:
            print(f"[ERROR] Failed to create model {model_name}: {e}")
            raise

    @staticmethod
    def freeze_backbone(model: nn.Module, verbose: bool = True) -> None:
        """
        Freeze backbone layers, keep classifier trainable.
        """
        classifier_keywords = ['fc', 'classifier', 'heads']
        
        frozen_count = 0
        trainable_count = 0
        
        for name, param in model.named_parameters():
            if not any(keyword in name for keyword in classifier_keywords):
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True
                trainable_count += param.numel()
        
        if verbose:
            print(f"[FREEZE] Frozen: {frozen_count:,}, Trainable: {trainable_count:,}")

    @staticmethod
    def unfreeze_backbone(model: nn.Module, verbose: bool = True) -> None:
        """
        Unfreeze all layers for fine-tuning.
        """
        for param in model.parameters():
            param.requires_grad = True
        
        if verbose:
            total = sum(p.numel() for p in model.parameters())
            print(f"[UNFREEZE] All {total:,} parameters now trainable")

    @staticmethod
    def get_model_summary(model: nn.Module) -> Dict[str, Any]:
        """
        Get detailed model summary.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_layers = sum(1 for _ in model.modules())
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params,
            'num_layers': num_layers,
            'model_size_mb': total_params * 4 / 1024 / 1024
        }

# ============================================================================
# ADVANCED TRAINING ENGINE - FIXED FOR MEMORY MANAGEMENT
# ============================================================================

class TrainingHistory:
    """
    Comprehensive training history tracker with GPU memory monitoring.
    """
    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'learning_rate': [],
            'time_per_epoch': [],
            'gpu_memory_mb': []
        }
        self.best_epoch = 0
        self.best_val_acc = 0.0
        
    def update(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float, time_taken: float):
        """Update history with new epoch data."""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_acc'].append(train_metrics['acc'])
        self.history['train_f1'].append(train_metrics.get('f1', 0))
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_acc'].append(val_metrics['acc'])
        self.history['val_f1'].append(val_metrics['f1'])
        self.history['val_precision'].append(val_metrics['precision'])
        self.history['val_recall'].append(val_metrics['recall'])
        self.history['learning_rate'].append(lr)
        self.history['time_per_epoch'].append(time_taken)
        
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**2
            self.history['gpu_memory_mb'].append(mem_allocated)
        
        if val_metrics['acc'] > self.best_val_acc:
            self.best_val_acc = val_metrics['acc']
            self.best_epoch = epoch
            
    def save(self, filepath: str):
        """Save history to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot(self, save_path: str):
        """Plot comprehensive training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        epochs = self.history['epoch']
        
        # Loss
        axes[0,0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0,0].plot(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0,0].set_title('Loss Curve', fontweight='bold')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best ({self.best_epoch})')
        
        # Accuracy
        axes[0,1].plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[0,1].plot(epochs, self.history['val_acc'], 'r-', label='Val', linewidth=2)
        axes[0,1].set_title('Accuracy Curve', fontweight='bold')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy (%)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.5)
        
        # F1 Score
        axes[0,2].plot(epochs, self.history['train_f1'], 'b-', label='Train', linewidth=2)
        axes[0,2].plot(epochs, self.history['val_f1'], 'r-', label='Val', linewidth=2)
        axes[0,2].set_title('F1 Score', fontweight='bold')
        axes[0,2].set_xlabel('Epoch')
        axes[0,2].set_ylabel('F1')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # Precision vs Recall
        axes[1,0].plot(epochs, self.history['val_precision'], 'g-', label='Precision', linewidth=2)
        axes[1,0].plot(epochs, self.history['val_recall'], 'm-', label='Recall', linewidth=2)
        axes[1,0].set_title('Precision vs Recall', fontweight='bold')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1,1].plot(epochs, self.history['learning_rate'], 'c-', linewidth=2)
        axes[1,1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('LR')
        axes[1,1].set_yscale('log')
        axes[1,1].grid(True, alpha=0.3)
        
        # GPU Memory
        if self.history['gpu_memory_mb']:
            axes[1,2].plot(epochs, self.history['gpu_memory_mb'], 'orange', linewidth=2)
            axes[1,2].axhline(y=12*1024*0.9, color='r', linestyle='--', alpha=0.5, label='90% of 12GB')
            axes[1,2].set_title('GPU Memory Usage (MB)', fontweight='bold')
            axes[1,2].set_xlabel('Epoch')
            axes[1,2].set_ylabel('Memory (MB)')
            axes[1,2].legend()
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

class AdvancedTrainer:
    """
    Advanced training manager with two-phase strategy, 
    mixed precision, and comprehensive logging.
    """
    
    def __init__(self, model: nn.Module, model_name: str, device: torch.device):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.scaler = GradScaler() if MIXED_PRECISION else None
        self.history = TrainingHistory()
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.early_stop_counter = 0
        self.current_phase = None
        
    def setup_phase(self, phase: str, learning_rate: float, weight_decay: float = 1e-4):
        """
        Setup optimizer and scheduler for training phase.
        """
        self.current_phase = phase
        
        if phase == 'warmup':
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            print(f"[SETUP] Warmup phase: Training classifier only | LR: {learning_rate}")
        else:
            backbone_params = []
            head_params = []
            
            for name, param in self.model.named_parameters():
                if any(key in name for key in ['fc', 'classifier', 'heads']):
                    head_params.append(param)
                else:
                    backbone_params.append(param)
            
            params = [
                {'params': backbone_params, 'lr': learning_rate * 0.1, 'weight_decay': weight_decay},
                {'params': head_params, 'lr': learning_rate, 'weight_decay': weight_decay}
            ]
            print(f"[SETUP] Fine-tune phase: Backbone LR={learning_rate*0.1:.2e}, Head LR={learning_rate:.2e}")
        
        self.optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
            verbose=True,
            min_lr=1e-7
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch with mixed precision.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device, non_blocking=False)  # FIXED: non_blocking=False for Windows
            labels = labels.to(self.device, non_blocking=False)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass
            if MIXED_PRECISION and self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels) / GRADIENT_ACCUMULATION_STEPS
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels) / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
                    self.optimizer.step()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        acc = 100. * correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        return {
            'loss': total_loss / len(dataloader),
            'acc': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate model and return metrics + predictions.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in dataloader:
            images = images.to(self.device, non_blocking=False)  # FIXED
            labels = labels.to(self.device, non_blocking=False)
            
            if MIXED_PRECISION and self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total_loss += loss.item()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        acc = 100. * correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'acc': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        return metrics, np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, 
            epochs: int, phase_name: str) -> None:
        """
        Full training loop with early stopping and comprehensive logging.
        """
        print(f"\n{'='*80}")
        print(f"TRAINING PHASE: {phase_name}")
        print(f"Model: {self.model_name} | Epochs: {epochs} | Mixed Precision: {MIXED_PRECISION}")
        print(f"{'='*80}\n")
        
        for epoch in range(epochs):
            start_time = time.time()
            epoch_start_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics, val_labels, val_preds, val_probs = self.validate(val_loader)
            
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - start_time
            epoch_max_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            self.history.update(epoch + 1, train_metrics, val_metrics, current_lr, epoch_time)
            
            print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | "
                  f"Time: {epoch_time:.1f}s | "
                  f"GPU: {epoch_start_mem:.1f}-{epoch_max_mem:.1f}GB | "
                  f"LR: {current_lr:.2e} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['acc']:.2f}% | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['acc']:.2f}% | "
                  f"Val F1: {val_metrics['f1']:.4f}")
            
            if val_metrics['acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc']
                self.best_val_f1 = val_metrics['f1']
                self.early_stop_counter = 0
                self.save_checkpoint(f"{self.model_name}_{self.current_phase}_best.pth")
                print(f"  >>> New best model saved (Val Acc: {val_metrics['acc']:.2f}%)")
            else:
                self.early_stop_counter += 1
            
            if self.early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n[WARNING] Early stopping triggered at epoch {epoch+1}")
                break
            
            # FIXED: Periodic memory cleanup
            if torch.cuda.is_available() and epoch % EMPTY_CACHE_FREQUENCY == 0:
                torch.cuda.empty_cache()
        
        print(f"\n[SUMMARY] Best Val Accuracy: {self.best_val_acc:.2f}% "
              f"(F1: {self.best_val_f1:.4f}) at epoch {self.history.best_epoch}")
        
        plot_path = os.path.join(OUTPUT_DIRS['plots_training'], 
                                f"{self.model_name}_{self.current_phase}_curves.png")
        self.history.plot(plot_path)
        
        history_path = os.path.join(OUTPUT_DIRS['logs'], 
                                   f"{self.model_name}_{self.current_phase}_history.json")
        self.history.save(history_path)
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint with metadata."""
        filepath = os.path.join(OUTPUT_DIRS['checkpoints'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history.history,
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'model_name': self.model_name,
            'phase': self.current_phase
        }, filepath)
    
    def load_best_checkpoint(self) -> None:
        """Load best checkpoint for current phase."""
        pattern = f"{self.model_name}_{self.current_phase}_best.pth"
        filepath = os.path.join(OUTPUT_DIRS['checkpoints'], pattern)
        
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[LOADED] Best model from {self.current_phase} phase "
                  f"(Acc: {checkpoint['best_val_acc']:.2f}%)")

# ============================================================================
# THRESHOLD OPTIMIZATION ENGINE - COMPLETE
# ============================================================================

class ThresholdOptimizer:
    """
    Advanced threshold tuning for multi-class classification.
    Implements Youden's J statistic, F1 optimization, and cost-sensitive thresholds.
    """
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.optimal_thresholds = {}
        self.threshold_methods = {}
        
    def find_optimal_thresholds(self, 
                               y_true: np.ndarray, 
                               y_probs: np.ndarray,
                               method: str = 'youden') -> Dict[str, float]:
        """
        Find optimal threshold for each class using specified method.
        """
        print(f"\n[THRESHOLD OPTIMIZATION] Method: {method}")
        
        y_true_bin = np.zeros((len(y_true), self.num_classes))
        for i, label in enumerate(y_true):
            y_true_bin[i, label] = 1
        
        thresholds = np.arange(0.05, 0.95, 0.01)
        optimal = {}
        
        for i, class_name in enumerate(self.class_names):
            best_thresh = 0.5
            best_score = -np.inf
            
            for thresh in thresholds:
                y_pred = (y_probs[:, i] >= thresh).astype(int)
                
                tp = np.sum((y_pred == 1) & (y_true_bin[:, i] == 1))
                tn = np.sum((y_pred == 0) & (y_true_bin[:, i] == 0))
                fp = np.sum((y_pred == 1) & (y_true_bin[:, i] == 0))
                fn = np.sum((y_pred == 0) & (y_true_bin[:, i] == 1))
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = sensitivity
                
                if method == 'youden':
                    score = sensitivity + specificity - 1
                elif method == 'f1':
                    score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                elif method == 'balanced':
                    score = (precision + recall) / 2
                elif method == 'sensitive':
                    score = sensitivity * 0.9 + specificity * 0.1
                elif method == 'specific':
                    score = sensitivity * 0.1 + specificity * 0.9
                else:
                    score = f1_score(y_true_bin[:, i], y_pred)
                
                if score > best_score:
                    best_score = score
                    best_thresh = thresh
            
            optimal[class_name] = float(best_thresh)
            self.threshold_methods[class_name] = method
            print(f"  {class_name:15s}: {best_thresh:.3f} (score: {best_score:.3f})")
        
        self.optimal_thresholds = optimal
        return optimal
    
    def apply_thresholds(self, y_probs: np.ndarray, thresholds: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Apply optimized thresholds to probabilities.
        """
        if thresholds is None:
            thresholds = self.optimal_thresholds
        
        thresh_array = np.array([thresholds.get(cls, 0.5) for cls in self.class_names])
        adjusted_scores = y_probs / thresh_array
        return np.argmax(adjusted_scores, axis=1)
    
    def plot_threshold_analysis(self, 
                               y_true: np.ndarray, 
                               y_probs: np.ndarray,
                               model_name: str) -> None:
        """
        Comprehensive threshold analysis visualization.
        """
        y_true_bin = np.zeros((len(y_true), self.num_classes))
        for i, label in enumerate(y_true):
            y_true_bin[i, label] = 1
        
        fig = plt.figure(figsize=(20, 15))
        
        # ROC Curves
        plt.subplot(3, 3, 1)
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2, 
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves', fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall Curves
        plt.subplot(3, 3, 2)
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_probs[:, i])
            plt.plot(recall, precision, color=colors[i], lw=2,
                    label=f'{class_name} (AP = {ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves', fontweight='bold')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        # Threshold vs F1 for each class
        thresholds = np.arange(0.1, 0.9, 0.02)
        
        for idx, (metric_name, metric_func) in enumerate([
            ('F1 Score', lambda yt, yp: precision_recall_fscore_support(yt, yp, average='binary', zero_division=0)[2]),
            ('Youden J', lambda yt, yp: (np.sum((yp==1)&(yt==1))/(np.sum((yp==1)&(yt==1))+np.sum((yp==0)&(yt==1))) if (np.sum((yp==1)&(yt==1))+np.sum((yp==0)&(yt==1))) > 0 else 0) + 
                                        (np.sum((yp==0)&(yt==0))/(np.sum((yp==0)&(yt==0))+np.sum((yp==1)&(yt==0))) if (np.sum((yp==0)&(yt==0))+np.sum((yp==1)&(yt==0))) > 0 else 0) - 1),
            ('Accuracy', lambda yt, yp: accuracy_score(yt, yp))
        ]):
            plt.subplot(3, 3, 3 + idx)
            for i, class_name in enumerate(self.class_names):
                scores = []
                for thresh in thresholds:
                    y_pred = (y_probs[:, i] >= thresh).astype(int)
                    score = metric_func(y_true_bin[:, i], y_pred)
                    scores.append(score)
                plt.plot(thresholds, scores, color=colors[i], marker='o', markersize=3, label=class_name)
            plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.3)
            plt.xlabel('Threshold')
            plt.ylabel(metric_name)
            plt.title(f'Threshold vs {metric_name}', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Calibration plot
        plt.subplot(3, 3, 6)
        for i, class_name in enumerate(self.class_names):
            prob_true, prob_pred = self._calibration_curve(y_true_bin[:, i], y_probs[:, i])
            plt.plot(prob_pred, prob_true, 's-', color=colors[i], label=class_name)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Optimal thresholds bar chart
        plt.subplot(3, 3, 7)
        thresholds_list = [self.optimal_thresholds.get(cls, 0.5) for cls in self.class_names]
        bars = plt.bar(self.class_names, thresholds_list, color=colors)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
        plt.ylabel('Optimal Threshold')
        plt.title('Optimal Thresholds by Class', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        for bar, thresh in zip(bars, thresholds_list):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{thresh:.2f}', ha='center', va='bottom')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Confusion matrix with optimal thresholds
        plt.subplot(3, 3, 8)
        tuned_preds = self.apply_thresholds(y_probs)
        cm = confusion_matrix(y_true, tuned_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix\n(Optimized Thresholds)', fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Metrics comparison table
        plt.subplot(3, 3, 9)
        plt.axis('off')
        
        default_preds = np.argmax(y_probs, axis=1)
        tuned_preds = self.apply_thresholds(y_probs)
        
        default_acc = accuracy_score(y_true, default_preds)
        tuned_acc = accuracy_score(y_true, tuned_preds)
        
        default_f1 = precision_recall_fscore_support(y_true, default_preds, average='weighted')[2]
        tuned_f1 = precision_recall_fscore_support(y_true, tuned_preds, average='weighted')[2]
        
        table_data = [
            ['Metric', 'Default (0.5)', 'Optimized', 'Improvement'],
            ['Accuracy', f'{default_acc:.4f}', f'{tuned_acc:.4f}', f'{tuned_acc-default_acc:+.4f}'],
            ['F1 Score', f'{default_f1:.4f}', f'{tuned_f1:.4f}', f'{tuned_f1-default_f1:+.4f}']
        ]
        
        table = plt.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.25, 0.25, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Performance Improvement', fontweight='bold', pad=20)
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIRS['plots_thresholds'], f"{model_name}_threshold_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        threshold_data = {
            'optimal_thresholds': self.optimal_thresholds,
            'methods': self.threshold_methods,
            'improvement': {
                'accuracy_default': float(default_acc),
                'accuracy_optimized': float(tuned_acc),
                'f1_default': float(default_f1),
                'f1_optimized': float(tuned_f1)
            }
        }
        
        with open(os.path.join(OUTPUT_DIRS['plots_thresholds'], f"{model_name}_thresholds.json"), 'w') as f:
            json.dump(threshold_data, f, indent=2)
    
    def _calibration_curve(self, y_true, y_prob, n_bins=10):
        """Helper for calibration plot."""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]
        
        bin_centers = []
        bin_accuracies = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracy_in_bin)
        
        return np.array(bin_accuracies), np.array(bin_centers)

# ============================================================================
# EMBEDDING EXTRACTOR & VISUALIZER - COMPLETE
# ============================================================================

class FeatureEmbeddingExtractor:
    """
    Extract feature embeddings from penultimate layer and visualize
    using dimensionality reduction techniques (t-SNE, UMAP, PCA).
    """
    
    def __init__(self, model: nn.Module, model_name: str, device: torch.device):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.embeddings = None
        self.labels = None
        self.images = None
        
    def extract_features(self, 
                        dataloader: DataLoader, 
                        max_samples: int = 1000,  # Reduced for memory
                        layer_type: str = 'penultimate') -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from specified layer.
        """
        print(f"\n[EMBEDDING] Extracting features from {self.model_name}...")
        self.model.eval()
        
        features = []
        labels = []
        images_list = []
        
        activation = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        handle = None
        if hasattr(self.model, 'fc'):
            if isinstance(self.model.fc, nn.Sequential):
                handle = self.model.fc[0].register_forward_hook(get_activation('feat'))
            else:
                handle = self.model.fc.register_forward_hook(get_activation('feat'))
        elif hasattr(self.model, 'classifier'):
            if isinstance(self.model.classifier, nn.Sequential):
                handle = self.model.classifier[0].register_forward_hook(get_activation('feat'))
            else:
                handle = self.model.classifier.register_forward_hook(get_activation('feat'))
        elif hasattr(self.model, 'heads'):
            if isinstance(self.model.heads.head, nn.Sequential):
                handle = self.model.heads.head[0].register_forward_hook(get_activation('feat'))
            else:
                handle = self.model.heads.head.register_forward_hook(get_activation('feat'))
        
        if handle is None:
            print("[WARNING] Could not register hook, skipping embedding extraction")
            return None, None
        
        count = 0
        with torch.no_grad():
            for images, lbls in dataloader:
                if count >= max_samples:
                    break
                    
                images = images.to(self.device)
                _ = self.model(images)
                
                if 'feat' in activation:
                    feat = activation['feat']
                    if len(feat.shape) > 2:
                        feat = feat.view(feat.size(0), -1)
                    
                    features.extend(feat.cpu().numpy())
                    labels.extend(lbls.numpy())
                    
                    if len(images_list) < 50:  # Reduced
                        images_list.extend(images.cpu())
                    
                    count += len(lbls)
        
        handle.remove()
        
        self.embeddings = np.array(features)
        self.labels = np.array(labels)
        self.images = images_list
        
        print(f"[EMBEDDING] Extracted {len(features)} embeddings of dimension {features[0].shape}")
        
        np.save(os.path.join(OUTPUT_DIRS['embeddings'], f"{self.model_name}_embeddings.npy"), self.embeddings)
        np.save(os.path.join(OUTPUT_DIRS['embeddings'], f"{self.model_name}_labels.npy"), self.labels)
        
        return self.embeddings, self.labels
    
    def visualize(self, 
                 class_names: List[str],
                 methods: List[str] = ['tsne', 'pca'],
                 perplexity: int = 30,
                 n_neighbors: int = 15) -> None:
        """
        Visualize embeddings using multiple dimensionality reduction methods.
        """
        if self.embeddings is None or len(self.embeddings) < 10:
            print("[WARNING] Not enough embeddings for visualization")
            return
        
        print(f"[EMBEDDING] Generating visualizations...")
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
        
        for method in methods:
            print(f"  Computing {method.upper()}...")
            
            try:
                if method == 'tsne':
                    reducer = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(self.embeddings)-1))
                    title_suffix = 't-SNE'
                elif method == 'pca':
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=2)
                    title_suffix = 'PCA'
                elif method == 'umap' and UMAP_AVAILABLE:
                    reducer = umap.UMAP(n_neighbors=min(n_neighbors, len(self.embeddings)-1), random_state=42)
                    title_suffix = 'UMAP'
                else:
                    continue
                
                embeddings_2d = reducer.fit_transform(self.embeddings)
                
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                
                ax = axes[0]
                for i, class_name in enumerate(class_names):
                    mask = self.labels == i
                    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                              c=[colors[i]], label=class_name, alpha=0.6, s=50, edgecolors='none')
                
                ax.legend(loc='best', fontsize=10)
                ax.set_title(f'{self.model_name} - {title_suffix} Visualization\n(colored by class)', 
                            fontsize=14, fontweight='bold')
                ax.set_xlabel(f'{title_suffix} 1')
                ax.set_ylabel(f'{title_suffix} 2')
                ax.grid(True, alpha=0.3)
                
                ax = axes[1]
                for i, class_name in enumerate(class_names):
                    mask = self.labels == i
                    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                              c=[colors[i]], alpha=0.3, s=20)
                
                from scipy.stats import gaussian_kde
                try:
                    xy = np.vstack([embeddings_2d[:, 0], embeddings_2d[:, 1]])
                    kde = gaussian_kde(xy)
                    
                    x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
                    y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
                    
                    x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
                    density = np.reshape(kde(positions).T, x_grid.shape)
                    
                    ax.contour(x_grid, y_grid, density, colors='k', alpha=0.3, levels=5)
                except Exception as e:
                    pass
                
                ax.set_title(f'{self.model_name} - {title_suffix} Density', 
                            fontsize=14, fontweight='bold')
                ax.set_xlabel(f'{title_suffix} 1')
                ax.set_ylabel(f'{title_suffix} 2')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                save_path = os.path.join(OUTPUT_DIRS['embeddings'], 
                                        f"{self.model_name}_{method}_visualization.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"  Error with {method}: {e}")
                continue

# ============================================================================
# COMPREHENSIVE EVALUATION SUITE - COMPLETE
# ============================================================================

class ModelEvaluator:
    """
    Comprehensive evaluation suite for trained models.
    Generates detailed metrics, confusion matrices, and per-class analysis.
    """
    
    def __init__(self, class_names: List[str], output_dirs: Dict[str, str]):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.output_dirs = output_dirs
        self.results_cache = {}
        
    def evaluate(self, 
                model: nn.Module,
                test_loader: DataLoader,
                model_name: str,
                threshold_optimizer: Optional[ThresholdOptimizer] = None) -> Dict[str, Any]:
        """
        Complete evaluation pipeline.
        """
        print(f"\n[EVALUATION] Evaluating {model_name} on test set...")
        
        model.eval()
        all_probs = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        print(f"\n{'='*60}")
        print(f"RESULTS FOR {model_name}")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"Kappa:     {metrics['cohen_kappa']:.4f}")
        print(f"MCC:       {metrics['mcc']:.4f}")
        
        self._plot_confusion_matrix(all_labels, all_preds, model_name)
        self._plot_per_class_metrics(all_labels, all_preds, all_probs, model_name)
        
        if threshold_optimizer:
            thresholds = threshold_optimizer.find_optimal_thresholds(all_labels, all_probs, method='youden')
            threshold_optimizer.plot_threshold_analysis(all_labels, all_probs, model_name)
            
            tuned_preds = threshold_optimizer.apply_thresholds(all_probs)
            tuned_metrics = self._calculate_metrics(all_labels, tuned_preds, all_probs)
            
            print(f"\nWith Optimized Thresholds:")
            print(f"Accuracy: {tuned_metrics['accuracy']:.4f} (Δ{tuned_metrics['accuracy']-metrics['accuracy']:+.4f})")
            print(f"F1-Score: {tuned_metrics['f1_score']:.4f} (Δ{tuned_metrics['f1_score']-metrics['f1_score']:+.4f})")
            
            metrics['tuned'] = tuned_metrics
            metrics['optimal_thresholds'] = thresholds
        
        self._save_detailed_report(all_labels, all_preds, all_probs, model_name, metrics)
        
        self.results_cache[model_name] = {
            'metrics': metrics,
            'probs': all_probs,
            'labels': all_labels,
            'preds': all_preds
        }
        
        return metrics
    
    def _calculate_metrics(self, 
                          y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        logloss = log_loss(y_true, y_prob) if y_prob is not None else None
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cohen_kappa': kappa,
            'mcc': mcc,
            'log_loss': logloss,
            'per_class': {
                'precision': per_class_precision.tolist(),
                'recall': per_class_recall.tolist(),
                'f1': per_class_f1.tolist()
            }
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot normalized and raw confusion matrices."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        cm_raw = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=self.class_names, yticklabels=self.class_names)
        axes[0].set_title(f'{model_name}\nConfusion Matrix (Counts)', fontweight='bold')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        cm_norm = cm_raw.astype('float') / cm_raw.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                   xticklabels=self.class_names, yticklabels=self.class_names)
        axes[1].set_title(f'{model_name}\nConfusion Matrix (Normalized)', fontweight='bold')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirs['plots_evaluation'], f"{model_name}_confusion_matrix.png"), dpi=300)
        plt.close()
    
    def _plot_per_class_metrics(self, y_true, y_pred, y_prob, model_name):
        """Plot per-class precision, recall, F1."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
        bars2 = ax.bar(x, recall, width, label='Recall', color='lightgreen')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='salmon')
        
        ax.set_ylabel('Score')
        ax.set_title(f'{model_name} - Per-Class Metrics', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirs['plots_evaluation'], f"{model_name}_per_class_metrics.png"), dpi=300)
        plt.close()
    
    def _save_detailed_report(self, y_true, y_pred, y_prob, model_name, metrics):
        """Save comprehensive JSON report."""
        report = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                k: v for k, v in metrics.items() if k not in ['per_class', 'tuned', 'optimal_thresholds']
            },
            'per_class_metrics': {
                class_name: {
                    'precision': metrics['per_class']['precision'][i],
                    'recall': metrics['per_class']['recall'][i],
                    'f1': metrics['per_class']['f1'][i]
                }
                for i, class_name in enumerate(self.class_names)
            },
            'classification_report': classification_report(
                y_true, y_pred, target_names=self.class_names, output_dict=True
            )
        }
        
        if 'tuned' in metrics:
            report['optimized_metrics'] = {
                'accuracy': metrics['tuned']['accuracy'],
                'f1_score': metrics['tuned']['f1_score']
            }
            report['optimal_thresholds'] = metrics.get('optimal_thresholds', {})
        
        with open(os.path.join(self.output_dirs['reports'], f"{model_name}_report.json"), 'w') as f:
            json.dump(report, f, indent=2)

# ============================================================================
# ENSEMBLE METHODS - COMPLETE
# ============================================================================

class EnsembleManager:
    """
    Advanced ensemble methods: Soft voting, hard voting, 
    weighted voting, and stacking.
    """
    
    def __init__(self, models_dict: Dict[str, nn.Module], device: torch.device):
        self.models = models_dict
        self.device = device
        self.weights = {name: 1.0 for name in models_dict.keys()}
        
    def set_weights(self, weights: Dict[str, float]):
        """Set custom weights for weighted voting."""
        self.weights = weights
    
    def predict(self, 
               dataloader: DataLoader, 
               method: str = 'soft') -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate ensemble predictions.
        """
        print(f"\n[ENSEMBLE] Generating predictions using {method} voting...")
        
        all_model_probs = {}
        all_labels = None
        
        for name, model in self.models.items():
            model.eval()
            probs_list = []
            
            with torch.no_grad():
                for images, labels in dataloader:
                    images = images.to(self.device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    probs_list.extend(probs.cpu().numpy())
                    
                    if all_labels is None:
                        all_labels = labels.numpy()
            
            all_model_probs[name] = np.array(probs_list)
            print(f"  ✓ {name}: collected predictions")
        
        if method == 'soft':
            ensemble_probs = np.mean(list(all_model_probs.values()), axis=0)
            
        elif method == 'weighted':
            total_weight = sum(self.weights.values())
            ensemble_probs = np.zeros_like(list(all_model_probs.values())[0])
            for name, probs in all_model_probs.items():
                ensemble_probs += probs * (self.weights[name] / total_weight)
                
        elif method == 'hard':
            all_preds = [np.argmax(probs, axis=1) for probs in all_model_probs.values()]
            all_preds = np.array(all_preds)
            
            ensemble_preds = np.zeros(all_preds.shape[1], dtype=int)
            for i in range(all_preds.shape[1]):
                votes = all_preds[:, i]
                ensemble_preds[i] = np.bincount(votes).argmax()
            
            return all_labels, ensemble_preds, all_model_probs
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        return all_labels, ensemble_preds, all_model_probs
    
    def evaluate(self, 
                dataloader: DataLoader, 
                class_names: List[str],
                methods: List[str] = ['soft', 'hard', 'weighted']) -> Dict[str, Dict]:
        """
        Evaluate ensemble using multiple methods.
        """
        results = {}
        
        for method in methods:
            print(f"\n[ENSEMBLE] Evaluating {method} voting...")
            labels, preds, individual_probs = self.predict(dataloader, method=method)
            
            acc = accuracy_score(labels, preds)
            p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
            
            results[method] = {
                'accuracy': acc,
                'precision': p,
                'recall': r,
                'f1_score': f1
            }
            
            print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        self._plot_ensemble_comparison(results, individual_probs, labels, class_names)
        
        return results
    
    def _plot_ensemble_comparison(self, 
                                 results: Dict, 
                                 individual_probs: Dict[str, np.ndarray],
                                 labels: np.ndarray,
                                 class_names: List[str]):
        """Visualize ensemble vs individual models."""
        fig = plt.figure(figsize=(20, 10))
        
        model_names = list(individual_probs.keys())
        accuracies = []
        
        for name, probs in individual_probs.items():
            preds = np.argmax(probs, axis=1)
            accuracies.append(accuracy_score(labels, preds))
        
        ensemble_names = list(results.keys())
        ensemble_accs = [results[m]['accuracy'] for m in ensemble_names]
        
        all_names = model_names + [f"Ensemble ({m})" for m in ensemble_names]
        all_accs = accuracies + ensemble_accs
        
        plt.subplot(2, 3, 1)
        colors = ['skyblue'] * len(model_names) + ['red', 'orange', 'green'][:len(ensemble_names)]
        bars = plt.bar(all_names, all_accs, color=colors)
        
        for bar, acc in zip(bars, all_accs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.ylabel('Accuracy')
        plt.title('Model Comparison: Individual vs Ensemble', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIRS['ensemble'], 'ensemble_comparison.png'), dpi=300)
        plt.close()
        
        with open(os.path.join(OUTPUT_DIRS['ensemble'], 'ensemble_results.json'), 'w') as f:
            json.dump({
                'individual': {name: float(acc) for name, acc in zip(model_names, accuracies)},
                'ensemble': {method: {k: float(v) for k, v in metrics.items()} 
                           for method, metrics in results.items()}
            }, f, indent=2)

# ============================================================================
# MAIN EXECUTION PIPELINE - FIXED WITH MEMORY MANAGEMENT
# ============================================================================

def get_optimal_batch_size(model_name: str, base_batch_size: int) -> int:
    """
    FIXED: Dynamically adjust batch size based on model size for 12GB GPU.
    """
    # Models that need smaller batches on 12GB
    large_models = ['efficientnet_b3', 'resnet101', 'vgg16', 'vit_b_16', 'vit_b_32', 'densenet161']
    very_large_models = ['efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                         'resnet152', 'densenet201', 'vit_l_16', 'vit_l_32']
    
    if model_name in very_large_models:
        return max(4, base_batch_size // 4)
    elif model_name in large_models:
        return max(8, base_batch_size // 2)
    return base_batch_size

def train_single_model(model_name: str,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      test_loader: DataLoader,
                      num_classes: int,
                      class_names: List[str],
                      batch_size: int) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Complete training pipeline for a single model.
    FIXED: Includes batch size parameter and memory cleanup.
    """
    print(f"\n{'#'*80}")
    print(f"# TRAINING: {model_name.upper()} (12GB Optimized)")
    print(f"{'#'*80}")
    
    # Clear cache before creating new model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    try:
        model = ArchitectureFactory.create_model(model_name, num_classes, dropout=DROPOUT_RATE)
        
        ArchitectureFactory.freeze_backbone(model)
        trainer = AdvancedTrainer(model, model_name, DEVICE)
        trainer.setup_phase('warmup', LEARNING_RATE_PHASE1, WEIGHT_DECAY)
        trainer.fit(train_loader, val_loader, EPOCHS_PHASE1, 'Warmup (Frozen Backbone)')
        
        ArchitectureFactory.unfreeze_backbone(model)
        trainer.setup_phase('finetune', LEARNING_RATE_PHASE2, WEIGHT_DECAY)
        trainer.fit(train_loader, val_loader, EPOCHS_PHASE2, 'Fine-tuning (Unfrozen)')
        
        trainer.load_best_checkpoint()
        
        evaluator = ModelEvaluator(class_names, OUTPUT_DIRS)
        threshold_opt = ThresholdOptimizer(class_names) if ENABLE_THRESHOLD_TUNING else None
        
        metrics = evaluator.evaluate(trainer.model, test_loader, model_name, threshold_opt)
        
        if ENABLE_EMBEDDING_EXTRACTION:
            extractor = FeatureEmbeddingExtractor(trainer.model, model_name, DEVICE)
            extractor.extract_features(test_loader, max_samples=1000)  # Reduced for memory
            extractor.visualize(class_names, methods=['tsne', 'pca'])
        
        return trainer.model, metrics
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[ERROR] Out of memory training {model_name}. Skipping...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, None
        raise e

def generate_final_report(all_results: List[Dict], class_names: List[str]):
    """
    Generate comprehensive final report comparing all models.
    """
    print(f"\n{'='*80}")
    print("FINAL COMPARISON REPORT")
    print(f"{'='*80}")
    
    # Filter out None results (failed models)
    all_results = [r for r in all_results if r is not None and r.get('metrics') is not None]
    
    if not all_results:
        print("[WARNING] No successful models to report!")
        return
    
    df_data = []
    for result in all_results:
        row = {
            'Model': result['model_name'],
            'Accuracy': f"{result['metrics']['accuracy']:.4f}",
            'Precision': f"{result['metrics']['precision']:.4f}",
            'Recall': f"{result['metrics']['recall']:.4f}",
            'F1-Score': f"{result['metrics']['f1_score']:.4f}",
            'Kappa': f"{result['metrics']['cohen_kappa']:.4f}",
            'MCC': f"{result['metrics']['mcc']:.4f}"
        }
        
        if 'tuned' in result['metrics']:
            row['Tuned_Acc'] = f"{result['metrics']['tuned']['accuracy']:.4f}"
            row['Tuned_F1'] = f"{result['metrics']['tuned']['f1_score']:.4f}"
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    
    csv_path = os.path.join(OUTPUT_DIRS['reports'], 'final_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVE] Comparison table saved to: {csv_path}")
    
    fig = plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 2, 1)
    models = df['Model'].tolist()
    accs = [float(a) for a in df['Accuracy'].tolist()]
    colors = ['red' if a == max(accs) else 'steelblue' for a in accs]
    
    bars = plt.bar(models, accs, color=colors)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([min(accs)-0.05, max(accs)+0.02])
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.subplot(2, 2, 2)
    f1s = [float(f) for f in df['F1-Score'].tolist()]
    colors = ['red' if f == max(f1s) else 'forestgreen' for f in f1s]
    plt.bar(models, f1s, color=colors)
    plt.ylabel('F1 Score')
    plt.title('Model F1-Score Comparison', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(2, 2, 3, projection='polar')
    best_idx = accs.index(max(accs))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'MCC']
    values = [
        float(df.iloc[best_idx]['Accuracy']),
        float(df.iloc[best_idx]['Precision']),
        float(df.iloc[best_idx]['Recall']),
        float(df.iloc[best_idx]['F1-Score']),
        float(df.iloc[best_idx]['Kappa']),
        float(df.iloc[best_idx]['MCC'])
    ]
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax = plt.subplot(2, 2, 3, projection='polar')
    ax.plot(angles, values, 'o-', linewidth=2, label=df.iloc[best_idx]['Model'])
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Best Model Performance\n(Radar Chart)', fontweight='bold', pad=20)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    if 'Tuned_Acc' in df.columns:
        plt.subplot(2, 2, 4)
        default_accs = [float(a) for a in df['Accuracy'].tolist()]
        tuned_accs = [float(a) for a in df['Tuned_Acc'].tolist()]
        improvements = [t - d for d, t in zip(default_accs, tuned_accs)]
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        plt.bar(models, improvements, color=colors, alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.ylabel('Accuracy Improvement')
        plt.title('Improvement with Threshold Tuning', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIRS['reports'], 'final_comparison_chart.png'), dpi=300)
    plt.close()
    
    with open(os.path.join(OUTPUT_DIRS['reports'], 'final_results.json'), 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'models': all_results,
            'best_model': models[best_idx],
            'best_accuracy': max(accs)
        }, f, indent=2)

def main():
    """
    Main execution pipeline.
    FIXED: Includes dynamic batch size adjustment and better error handling.
    """
    print(f"\n{'='*80}")
    print("STARTING BRAIN TUMOR CLASSIFICATION PIPELINE - 12GB GPU MODE")
    print(f"{'='*80}")
    print(f"Output Directory: {OUTPUT_BASE_PATH}")
    print(f"Models to train: {MODELS_TO_TRAIN}")
    print(f"Device: {DEVICE}")
    print(f"Configuration: Batch={BATCH_SIZE}, Image={IMAGE_SIZE}, Workers={NUM_WORKERS}")
    print(f"Windows Mode: NUM_WORKERS=0 (no multiprocessing)")
    
    # Create data loaders with default batch size
    train_loader, val_loader, test_loader, class_names, num_classes = create_data_loaders(
        TRAIN_DATA_PATH, TEST_DATA_PATH, BATCH_SIZE, VALIDATION_SPLIT, IMAGE_SIZE, NUM_WORKERS
    )
    
    trained_models = {}
    all_results = []
    
    for model_name in MODELS_TO_TRAIN:
        try:
            # FIXED: Get optimal batch size for this specific model
            current_batch = get_optimal_batch_size(model_name, BATCH_SIZE)
            
            if current_batch != BATCH_SIZE:
                print(f"\n[INFO] Adjusting batch size to {current_batch} for {model_name}")
                # Recreate data loaders with new batch size
                train_loader, val_loader, test_loader, _, _ = create_data_loaders(
                    TRAIN_DATA_PATH, TEST_DATA_PATH, current_batch, VALIDATION_SPLIT, IMAGE_SIZE, NUM_WORKERS
                )
            
            model, metrics = train_single_model(
                model_name, train_loader, val_loader, test_loader, num_classes, class_names, current_batch
            )
            
            if model is not None and metrics is not None:
                trained_models[model_name] = model
                all_results.append({
                    'model_name': model_name,
                    'metrics': metrics
                })
                
                # FIXED: Aggressive memory cleanup after each model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"[MEMORY] Cache cleared after {model_name}")
                    
        except Exception as e:
            print(f"\n[ERROR] Failed to train {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("\n[CRITICAL] No models trained successfully!")
        return
    
    generate_final_report(all_results, class_names)
    
    if ENABLE_ENSEMBLE and len(trained_models) > 1:
        print(f"\n{'='*80}")
        print("ENSEMBLE EVALUATION")
        print(f"{'='*80}")
        
        ensemble = EnsembleManager(trained_models, DEVICE)
        ensemble_results = ensemble.evaluate(test_loader, class_names, methods=['soft', 'weighted'])
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"All results saved to: {OUTPUT_BASE_PATH}")
    print(f"Subdirectories:")
    for key, path in OUTPUT_DIRS.items():
        print(f"  {key:20s}: {path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
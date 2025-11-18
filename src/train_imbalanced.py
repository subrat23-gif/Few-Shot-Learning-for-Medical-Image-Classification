import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchxrayvision as xrv
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from collections import Counter
import warnings

# --- 0. Configuration ---
# This is the folder you just created with split_data.py
SPLIT_DATA_DIR = 'data_split/' 

# Training Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = 'best_imbalanced_model.pth'

# ==============================================================================
# STAGE 2: IMBALANCED-AWARE TRAINING
# ==============================================================================

def get_xrv_transforms():
    """
    Get the specific transforms for torchxrayvision models (1-channel).
    """
    # This is the normalization specified by torchxrayvision
    XRV_MEAN = [0.5081]
    XRV_STD = [0.0893]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1), # XRV models expect 1 channel
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=XRV_MEAN, std=XRV_STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=XRV_MEAN, std=XRV_STD)
    ])
    
    return train_transform, val_test_transform

class XRVTransferModel(nn.Module):
    """
    A wrapper for the torchxrayvision model to add a custom classifier head.
    """
    def __init__(self, num_classes):
        super(XRVTransferModel, self).__init__()
        # Load pre-trained backbone
        # This model is pre-trained on chest x-rays, which is perfect
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.backbone = model.features
        
        # --- Freeze the backbone ---
        # We will only train the final classifier layer
        for name, param in self.backbone.named_parameters():
              if 'denseblock4' not in name:
                      param.requires_grad = False

             
        # Add a new classifier head
        # DenseNet-121 output is 1024 features
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # Regularization
            nn.Linear(1024, num_classes) # Output layer
        )

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.pooling(features).view(features.size(0), -1)
        output = self.classifier(pooled)
        return output

def create_weighted_sampler(dataset):
    """
    Creates a WeightedRandomSampler to handle class imbalance.
    """
    print("\nCalculating sampler weights for training set...")
    class_counts = Counter(dataset.targets)
    
    # Sort counts by class index (0, 1, 2, 3...)
    class_counts = [class_counts.get(i, 0) for i in range(len(dataset.classes))]
    print(f"  Class counts: {class_counts}")
    
    # Calculate weight per class (1 / num_samples)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    print(f"  Class weights: {class_weights}")
    
    # Assign a weight to every single sample in the dataset
    sample_weights = [class_weights[label] for label in dataset.targets]
    
    # Create the sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True # Oversampling: draw samples with replacement
    )
    print("WeightedRandomSampler created.")
    return sampler

def run_training():
    """
    Main function to run the entire training and evaluation pipeline.
    """
    print("--- Starting Imbalanced-Aware Training ---")
    
    # --- 1. Load Datasets ---
    train_transform, val_test_transform = get_xrv_transforms()
    
    train_dataset = ImageFolder(os.path.join(SPLIT_DATA_DIR, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(SPLIT_DATA_DIR, 'val'), transform=val_test_transform)
    test_dataset = ImageFolder(os.path.join(SPLIT_DATA_DIR, 'test'), transform=val_test_transform)
    
    # Get class names
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"\nTraining on {num_classes} classes: {class_names}")

    # --- 2. Create DataLoaders (with Sampler for train) ---
    train_sampler = create_weighted_sampler(train_dataset)
    
    # Note: shuffle=False because the sampler handles shuffling.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler, 
        num_workers=2
    )
    
    # Val and Test loaders should NOT be balanced.
    # We want to test on the real, imbalanced distribution.
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2
    )

    # --- 3. Initialize Model, Loss, Optimizer ---
    model = XRVTransferModel(num_classes=num_classes).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    # We are only training the head, as we froze the backbone
    optimizer = optim.Adam(
       filter(lambda p: p.requires_grad, model.parameters()), 
       lr=LEARNING_RATE
    )

    # --- 4. Training & Validation Loop ---
    best_val_f1 = 0.0
    
    # Suppress zero-division warnings from sklearn
    warnings.filterwarnings('ignore', category=UserWarning, message='F-score is ill-defined')

    print(f"\nStarting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        
        # CRITICAL: Use a weighted F1-score, not accuracy
        # 'average='weighted'' accounts for class imbalance in the F1 score
        val_f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val F1 (Weighted): {val_f1_weighted:.4f}")
        
        # Save the best model based on F1 score
        if val_f1_weighted > best_val_f1:
            best_val_f1 = val_f1_weighted
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  New best model saved to {MODEL_SAVE_PATH} (F1: {best_val_f1:.4f})")
            
    print("\nTraining complete.")

    # --- 5. Final Evaluation on Test Set ---
    print("\n--- FINAL EVALUATION ON TEST SET ---")
    
    # Load the *best* model we saved during training
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Loaded best model from {MODEL_SAVE_PATH}")
    except FileNotFoundError:
        print("Warning: No best model was saved. Evaluating last epoch model.")
        
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    # Print the final report
    print("\nClassification Report (Test Set):")
    # This report is the most important output.
    # It shows the precision, recall, and f1-score for EACH class.
    print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))
    
    print("\nConfusion Matrix (Test Set):")
    # Rows are (True Label), Columns are (Predicted Label)
    print(confusion_matrix(all_preds, all_targets))



if __name__ == "__main__":
    run_training()
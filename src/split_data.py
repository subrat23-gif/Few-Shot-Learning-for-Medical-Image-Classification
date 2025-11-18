import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# --- 1. Configuration ---
SOURCE_DATA_DIR = 'processed_data/'
TARGET_DATA_DIR = 'data_split/'
VAL_SIZE = 0.1
TEST_SIZE = 0.1
RANDOM_STATE = 42

# --- 2. The Splitting Function ---
def create_stratified_split(source_dir, target_dir):
    print(f"--- Starting Stratified Split ---")
    
    if os.path.exists(target_dir):
        print(f"Removing existing directory: {target_dir}")
        shutil.rmtree(target_dir)

    all_filepaths = []
    all_labels = []
    
    classes = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    if not classes:
        print(f"Error: No subdirectories found in {source_dir}. Aborting.")
        return False
        
    print(f"Found {len(classes)} classes: {classes}")

    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            if os.path.isfile(img_path):
                all_filepaths.append(img_path)
                all_labels.append(class_to_idx[cls])

    print(f"\nTotal images found: {len(all_filepaths)}")
    
    if not all_filepaths:
        print("Error: No image files found. Check your 'processed_data' folder.")
        return False
        
    print(f"Original Class distribution: {Counter([classes[i] for i in all_labels])}")

    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_filepaths, all_labels, 
        test_size=TEST_SIZE, 
        stratify=all_labels, 
        random_state=RANDOM_STATE
    )

    val_size_adjusted = VAL_SIZE / (1.0 - TEST_SIZE)
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, 
        test_size=val_size_adjusted, 
        stratify=train_val_labels, 
        random_state=RANDOM_STATE
    )
    
    print(f"\nSplitting data into:")
    print(f"  Train set: {len(train_files)} images")
    print(f"  Val set:   {len(val_files)} images")
    print(f"  Test set:  {len(test_files)} images")

    datasets = {
        'train': (train_files, train_labels),
        'val': (val_files, val_labels),
        'test': (test_files, test_labels)
    }

    for split_name, (files, labels) in datasets.items():
        split_path = os.path.join(target_dir, split_name)
        
        for i, filepath in enumerate(files):
            class_name = classes[labels[i]]
            target_class_dir = os.path.join(split_path, class_name)
            os.makedirs(target_class_dir, exist_ok=True)
            shutil.copy(filepath, target_class_dir)
            
    print(f"\nSuccessfully created stratified split at: {target_dir}")
    print("--- Data Splitting Complete ---")
    return True

# --- 3. Run the Function ---
if __name__ == "__main__":
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Error: scikit-learn is not installed.")
        print("Please install it by running: pip install scikit-learn")
        exit()

    if not os.path.exists(SOURCE_DATA_DIR):
        print(f"Error: Source directory not found: {SOURCE_DATA_DIR}")
        print("Please make sure your data is in a folder named 'processed_data' in the same directory.")
    else:
        create_stratified_split(SOURCE_DATA_DIR, TARGET_DATA_DIR)
"""
Preprocessing script for the Yoga Pose Checker project.
Handles data loading, splitting, augmentation, and preparation.
"""

import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm
from PIL import Image

# Import utility functions
from utils import (
    setup_processed_directories, create_train_val_split,
    apply_augmentation, preprocess_image, visualize_processed_samples,
    visualize_class_distribution, calculate_statistics, save_results_to_yaml
)

# Import configuration
from config import (
    DATA_DIR, PROCESSED_DIR, TRAIN_DIR, TEST_DIR, CLASSES,
    IMAGE_SIZE, VALIDATION_SPLIT, RANDOM_SEED, RESULTS_DIR
)

def count_dataset_images():
    """
    Count images in the original dataset by class.
    
    Returns:
        Dictionary with image counts by class and split
    """
    dataset_counts = {
        "train": {},
        "test": {}
    }
    
    # Count training images
    for cls in CLASSES:
        train_class_dir = os.path.join(TRAIN_DIR, cls)
        if os.path.exists(train_class_dir):
            images = [f for f in os.listdir(train_class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            dataset_counts["train"][cls] = len(images)
        else:
            # Create empty entry for missing class
            dataset_counts["train"][cls] = 0
            print(f"Warning: Class directory {train_class_dir} does not exist")
    
    # Count test images
    for cls in CLASSES:
        test_class_dir = os.path.join(TEST_DIR, cls)
        if os.path.exists(test_class_dir):
            images = [f for f in os.listdir(test_class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            dataset_counts["test"][cls] = len(images)
        else:
            # Create empty entry for missing class
            dataset_counts["test"][cls] = 0
            print(f"Warning: Class directory {test_class_dir} does not exist")
    
    return dataset_counts

def process_and_save_image(src_path, dest_path, target_size=IMAGE_SIZE):
    """
    Process a single image and save to destination.
    
    Args:
        src_path: Source image path
        dest_path: Destination path
        target_size: Target image size for resizing
        
    Returns:
        Boolean indicating success
    """
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Process image
        img_array = preprocess_image(src_path, target_size)
        if img_array is None:
            return False
        
        # Convert back to uint8 for saving
        img_uint8 = (img_array * 255).astype(np.uint8)
        
        # Save processed image
        cv2.imwrite(dest_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        return True
    
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

def process_dataset():
    """
    Process the yoga pose dataset:
    1. Split training data into train/validation sets
    2. Resize and normalize images
    3. Apply data augmentation to training set
    4. Save processed images
    
    Returns:
        Dictionary of processing statistics
    """
    # Check if all required directories exist
    setup_processed_directories()
    
    # Check original data directories
    for cls in CLASSES:
        train_cls_dir = os.path.join(TRAIN_DIR, cls)
        test_cls_dir = os.path.join(TEST_DIR, cls)
        
        if not os.path.exists(train_cls_dir):
            print(f"Error: Training directory {train_cls_dir} does not exist")
        
        if not os.path.exists(test_cls_dir):
            print(f"Error: Test directory {test_cls_dir} does not exist")
    
    # Split training data into train and validation
    train_files, val_files = create_train_val_split()
    
    # Tracking statistics
    processed_counts = {
        "train": {cls: 0 for cls in CLASSES},
        "val": {cls: 0 for cls in CLASSES},
        "test": {cls: 0 for cls in CLASSES},
        "augmented": {cls: 0 for cls in CLASSES}
    }
    
    # Process training files
    print("\nProcessing training files...")
    for src_path, cls in tqdm(train_files, desc="Training data"):
        # Extract filename
        filename = os.path.basename(src_path)
        dest_path = os.path.join(PROCESSED_DIR, "train", cls, filename)
        
        # Process and save
        if process_and_save_image(src_path, dest_path):
            processed_counts["train"][cls] += 1
            
            # Apply augmentation
            img_array = preprocess_image(src_path)
            if img_array is not None:
                augmented_images = apply_augmentation(img_array)
                
                # Save augmented images
                for aug_type, aug_img in augmented_images:
                    base_name = os.path.splitext(filename)[0]
                    aug_filename = f"{aug_type}_{base_name}.jpg"
                    aug_path = os.path.join(PROCESSED_DIR, "train", cls, aug_filename)
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(aug_path), exist_ok=True)
                    
                    # Convert to uint8 and save
                    aug_uint8 = (aug_img * 255).astype(np.uint8)
                    cv2.imwrite(aug_path, cv2.cvtColor(aug_uint8, cv2.COLOR_RGB2BGR))
                    processed_counts["augmented"][cls] += 1
    
    # Process validation files
    print("\nProcessing validation files...")
    for src_path, cls in tqdm(val_files, desc="Validation data"):
        filename = os.path.basename(src_path)
        dest_path = os.path.join(PROCESSED_DIR, "val", cls, filename)
        
        if process_and_save_image(src_path, dest_path):
            processed_counts["val"][cls] += 1
    
    # Process test files
    print("\nProcessing test files...")
    for cls in CLASSES:
        test_class_dir = os.path.join(TEST_DIR, cls)
        if os.path.exists(test_class_dir):
            for filename in tqdm(os.listdir(test_class_dir), desc=f"Test - {cls}"):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_path = os.path.join(test_class_dir, filename)
                    dest_path = os.path.join(PROCESSED_DIR, "test", cls, filename)
                    
                    if process_and_save_image(src_path, dest_path):
                        processed_counts["test"][cls] += 1
    
    # Verify directory structure
    verify_processed_dirs()
    
    return processed_counts

def verify_processed_dirs():
    """Verify all necessary processed directories exist."""
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            class_dir = os.path.join(PROCESSED_DIR, split, cls)
            if not os.path.exists(class_dir):
                print(f"Creating missing directory: {class_dir}")
                os.makedirs(class_dir, exist_ok=True)

def main():
    """Main function to preprocess the yoga poses dataset."""
    print("=" * 50)
    print("YOGA POSE DATASET PREPROCESSING")
    print("=" * 50)
    
    # Check if data exists
    if not (os.path.exists(TRAIN_DIR) and os.path.exists(TEST_DIR)):
        print("Error: Dataset not found. Please ensure the yoga poses dataset is in the data directory")
        print(f"Expected: {TRAIN_DIR} and {TEST_DIR}")
        return
    
    # Check if processed data already exists
    train_processed = os.path.exists(os.path.join(PROCESSED_DIR, "train"))
    val_processed = os.path.exists(os.path.join(PROCESSED_DIR, "val"))
    test_processed = os.path.exists(os.path.join(PROCESSED_DIR, "test"))
    
    if train_processed and val_processed and test_processed:
        print("\nProcessed data already exists.")
        
        # Verify all class directories exist
        verify_processed_dirs()
        
        # Calculate and save statistics for existing data
        print("\nCalculating dataset statistics...")
        stats_df = calculate_statistics(PROCESSED_DIR)
        print("\nExisting dataset distribution:")
        print(stats_df)
        
        # Visualize class distribution (save only, don't show)
        class_counts = {cls: stats_df["train"][cls] + stats_df["val"][cls] + stats_df["test"][cls] 
                        for cls in CLASSES}
        visualize_class_distribution(class_counts, "Yoga Poses Dataset Distribution",
                                   save_path=os.path.join(RESULTS_DIR, "class_distribution.png"))
        
        # Visualize sample processed images (save only, don't show)
        visualize_processed_samples(PROCESSED_DIR, 
                                  save_path=os.path.join(RESULTS_DIR, "processed_samples.png"))
                                  
        print(f"\nVisualization saved to {RESULTS_DIR}")
        return
    
    # Count original dataset images
    print("\nCounting dataset images...")
    dataset_counts = count_dataset_images()
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print("Training set:")
    train_total = sum(dataset_counts["train"].values())
    for cls, count in dataset_counts["train"].items():
        print(f"  {cls}: {count} images")
    print(f"  Total: {train_total} images")
    
    print("Test set:")
    test_total = sum(dataset_counts["test"].values())
    for cls, count in dataset_counts["test"].items():
        print(f"  {cls}: {count} images")
    print(f"  Total: {test_total} images")
    
    # Process dataset
    print("\nStarting dataset preprocessing...")
    processed_counts = process_dataset()
    
    # Verify all directories exist after processing
    verify_processed_dirs()
    
    # Calculate total processed counts
    train_processed = sum(processed_counts["train"].values())
    val_processed = sum(processed_counts["val"].values())
    test_processed = sum(processed_counts["test"].values())
    augmented = sum(processed_counts["augmented"].values())
    
    # Print processing results
    print("\nPreprocessing Results:")
    print(f"Training images: {train_processed}")
    print(f"Validation images: {val_processed}")
    print(f"Test images: {test_processed}")
    print(f"Augmented images: {augmented}")
    print(f"Total processed: {train_processed + val_processed + test_processed + augmented}")
    
    # Calculate full dataset statistics
    print("\nCalculating dataset statistics...")
    stats_df = calculate_statistics(PROCESSED_DIR)
    print("\nFinal dataset distribution:")
    print(stats_df)
    
    # Visualize sample processed images (save only, don't show)
    print("\nGenerating visualization samples...")
    visualize_processed_samples(PROCESSED_DIR, 
                              save_path=os.path.join(RESULTS_DIR, "processed_samples.png"))
    
    # Visualize class distribution (save only, don't show)
    class_counts = {cls: stats_df["train"][cls] + stats_df["val"][cls] + stats_df["test"][cls] 
                    for cls in CLASSES}
    visualize_class_distribution(class_counts, "Yoga Poses Dataset Distribution",
                               save_path=os.path.join(RESULTS_DIR, "class_distribution.png"))
    
    # Save preprocessing results
    preprocessing_results = {
        "original_counts": dataset_counts,
        "processed_counts": processed_counts,
        "final_distribution": stats_df.to_dict()
    }
    save_results_to_yaml(preprocessing_results, 
                       os.path.join(RESULTS_DIR, "preprocessing_results.yaml"))
    
    print(f"\nVisualizations saved to {RESULTS_DIR}")
    print("\nPreprocessing complete! Dataset ready for pose extraction.")

if __name__ == "__main__":
    main()
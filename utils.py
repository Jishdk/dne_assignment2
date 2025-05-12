"""
Utility functions for the Yoga Pose Checker project.
Contains helper functions used throughout the pipeline.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import pickle
import yaml
from sklearn.model_selection import train_test_split


from config import (
    CLASSES, PROCESSED_DIR, TRAIN_DIR, TEST_DIR, KEYPOINTS_DIR, VISUALIZATIONS_DIR,
    VALIDATION_SPLIT, RANDOM_SEED, IMAGE_SIZE, ESSENTIAL_KEYPOINTS, KEYPOINTS_INDEX
)


# ===================================
# Directory Management
# ===================================

def create_directory_structure(base_dirs, class_dirs=None):
    """Create a directory structure for the project."""
    # Create base directories
    for directory in base_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Create class subdirectories if needed
    if class_dirs:
        for base_dir in base_dirs:
            for sub_dir in class_dirs:
                full_path = os.path.join(base_dir, sub_dir)
                if not os.path.exists(full_path):
                    os.makedirs(full_path)

def setup_processed_directories():
    """Set up directories for processed images."""
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    
    # Create train/val/test directories with class subdirectories
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(PROCESSED_DIR, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        
        for cls in CLASSES:
            class_dir = os.path.join(split_dir, cls)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

def setup_keypoints_directories():
    """Set up directories for keypoints and visualizations."""
    for directory in [KEYPOINTS_DIR, VISUALIZATIONS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Create subdirectories for each split and class
    for directory in [KEYPOINTS_DIR, VISUALIZATIONS_DIR]:
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(directory, split)
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)
            
            for cls in CLASSES:
                class_dir = os.path.join(split_dir, cls)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

# ===================================
# Data Preprocessing
# ===================================

def preprocess_image(img_path, target_size=IMAGE_SIZE):
    """Preprocess a single image for model input."""
    try:
        # Load and convert image to RGB
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(target_size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
        
        # Normalize pixel values to [0, 1]
        img_array = np.array(img) / 255.0
        
        return img_array
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def create_train_val_split():
    """Split training data into train and validation sets."""
    train_files = []
    
    # Collect all training files with class labels
    for cls in CLASSES:
        class_dir = os.path.join(TRAIN_DIR, cls)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                train_files.append((os.path.join(class_dir, img_name), cls))
    
    # Shuffle and split (stratified by class)
    random.seed(RANDOM_SEED)
    random.shuffle(train_files)
    
    train_files_split, val_files = train_test_split(
        train_files, 
        test_size=VALIDATION_SPLIT, 
        random_state=RANDOM_SEED,
        stratify=[f[1] for f in train_files]
    )
    
    return train_files_split, val_files

def apply_augmentation(img_array):
    """Apply data augmentation (flipping, rotation) to an image."""
    augmented_images = []
    
    # Horizontal flip
    flip_img = img_array[:, ::-1, :]
    augmented_images.append(("flip", flip_img))
    
    # Random rotation between -15 and 15 degrees
    angle = random.uniform(-15, 15)
    h, w = img_array.shape[:2]
    center = (w/2, h/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_img = cv2.warpAffine(img_array, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
    augmented_images.append(("rot", rot_img))
    
    return augmented_images

# ===================================
# Keypoint Processing
# ===================================

def normalize_keypoints(keypoints, image_size):
    """Normalize keypoint coordinates to [0, 1] range."""
    normalized = keypoints.copy()
    normalized[:, 0] /= image_size[0]  # x-coordinates
    normalized[:, 1] /= image_size[1]  # y-coordinates
    return normalized

def keypoints_to_features(keypoints):
    """Convert keypoints to feature vector for classification."""
    return keypoints.flatten()

def calculate_joint_angle(a, b, c):
    """Calculate angle between three points (joint angle) in degrees."""
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Compute cosine of angle
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = np.clip(cosine, -1.0, 1.0)  # Prevent numerical errors
    
    # Convert to degrees
    angle = np.arccos(cosine) * 180.0 / np.pi
    
    return angle

def extract_pose_angles(keypoints):
    """
    Extract important joint angles from MediaPipe keypoints.
    
    Args:
        keypoints (numpy.ndarray): MediaPipe keypoints array [33, 3]
        
    Returns:
        dict: Dictionary of joint angles
    """
    # Skip if any essential keypoint is missing or has low confidence
    keypoints_xy = keypoints[:, :2]  # Extract just x, y coordinates
    keypoints_conf = keypoints[:, 2]  # Extract confidence/visibility
    
    if np.any(keypoints_conf[ESSENTIAL_KEYPOINTS] < 0.5):
        return None
    
    # Dictionary to store angles
    angles = {}
    
    # Shoulder angles (nose-shoulder-elbow)
    angles['left_shoulder'] = calculate_joint_angle(
        keypoints_xy[KEYPOINTS_INDEX['nose']],  # nose
        keypoints_xy[KEYPOINTS_INDEX['left_shoulder']],  # left_shoulder
        keypoints_xy[KEYPOINTS_INDEX['left_elbow']]   # left_elbow
    )
    
    angles['right_shoulder'] = calculate_joint_angle(
        keypoints_xy[KEYPOINTS_INDEX['nose']],  # nose
        keypoints_xy[KEYPOINTS_INDEX['right_shoulder']],  # right_shoulder
        keypoints_xy[KEYPOINTS_INDEX['right_elbow']]   # right_elbow
    )
    
    # Elbow angles (shoulder-elbow-wrist)
    angles['left_elbow'] = calculate_joint_angle(
        keypoints_xy[KEYPOINTS_INDEX['left_shoulder']],  # left_shoulder
        keypoints_xy[KEYPOINTS_INDEX['left_elbow']],  # left_elbow
        keypoints_xy[KEYPOINTS_INDEX['left_wrist']]   # left_wrist
    )
    
    angles['right_elbow'] = calculate_joint_angle(
        keypoints_xy[KEYPOINTS_INDEX['right_shoulder']],  # right_shoulder
        keypoints_xy[KEYPOINTS_INDEX['right_elbow']],  # right_elbow
        keypoints_xy[KEYPOINTS_INDEX['right_wrist']]  # right_wrist
    )
    
    # Hip angles (shoulder-hip-knee)
    angles['left_hip'] = calculate_joint_angle(
        keypoints_xy[KEYPOINTS_INDEX['left_shoulder']],   # left_shoulder
        keypoints_xy[KEYPOINTS_INDEX['left_hip']],  # left_hip
        keypoints_xy[KEYPOINTS_INDEX['left_knee']]   # left_knee
    )
    
    angles['right_hip'] = calculate_joint_angle(
        keypoints_xy[KEYPOINTS_INDEX['right_shoulder']],   # right_shoulder
        keypoints_xy[KEYPOINTS_INDEX['right_hip']],  # right_hip
        keypoints_xy[KEYPOINTS_INDEX['right_knee']]   # right_knee
    )
    
    # Knee angles (hip-knee-ankle)
    angles['left_knee'] = calculate_joint_angle(
        keypoints_xy[KEYPOINTS_INDEX['left_hip']],  # left_hip
        keypoints_xy[KEYPOINTS_INDEX['left_knee']],  # left_knee
        keypoints_xy[KEYPOINTS_INDEX['left_ankle']]   # left_ankle
    )
    
    angles['right_knee'] = calculate_joint_angle(
        keypoints_xy[KEYPOINTS_INDEX['right_hip']],  # right_hip
        keypoints_xy[KEYPOINTS_INDEX['right_knee']],  # right_knee
        keypoints_xy[KEYPOINTS_INDEX['right_ankle']]   # right_ankle
    )
    
    return angles

# ===================================
# Visualization
# ===================================

def visualize_class_distribution(class_counts, title, save_path=None):
    """Plot the distribution of images across classes."""
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title(title)
    plt.xlabel("Yoga Pose Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    for i, v in enumerate(class_counts.values()):
        plt.text(i, v + 5, str(v), ha='center')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 

def visualize_processed_samples(processed_dir, save_path=None):
    """Show sample processed and augmented images from each class."""
    plt.figure(figsize=(15, 10))
    
    for i, cls in enumerate(CLASSES):
        # Get a random original image
        train_dir = os.path.join(processed_dir, "train", cls)
        train_samples = [f for f in os.listdir(train_dir) if not f.startswith(("flip_", "rot_"))]
        
        if train_samples:
            sample_path = os.path.join(train_dir, random.choice(train_samples))
            img = plt.imread(sample_path)
            
            plt.subplot(len(CLASSES), 3, i*3 + 1)
            plt.imshow(img)
            plt.title(f"{cls} - Original")
            plt.axis('off')
            
            # Get augmented images if available
            flip_samples = [f for f in os.listdir(train_dir) if f.startswith("flip_")]
            rot_samples = [f for f in os.listdir(train_dir) if f.startswith("rot_")]
            
            if flip_samples:
                flip_path = os.path.join(train_dir, random.choice(flip_samples))
                flip_img = plt.imread(flip_path)
                plt.subplot(len(CLASSES), 3, i*3 + 2)
                plt.imshow(flip_img)
                plt.title(f"{cls} - Flipped")
                plt.axis('off')
            
            if rot_samples:
                rot_path = os.path.join(train_dir, random.choice(rot_samples))
                rot_img = plt.imread(rot_path)
                plt.subplot(len(CLASSES), 3, i*3 + 3)
                plt.imshow(rot_img)
                plt.title(f"{cls} - Rotated")
                plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_keypoint_samples(visualization_dir, save_path=None):
    """Show sample pose estimations from each class."""
    plt.figure(figsize=(15, 10))
    
    for i, cls in enumerate(CLASSES):
        class_dir = os.path.join(visualization_dir, "train", cls)
        if os.path.exists(class_dir):
            vis_files = [f for f in os.listdir(class_dir) if f.endswith("_pose.jpg")]
            
            if vis_files:
                vis_path = os.path.join(class_dir, random.choice(vis_files))
                img = plt.imread(vis_path)
                
                plt.subplot(len(CLASSES), 1, i+1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(f"{cls} - Pose Estimation")
                plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

# ===================================
# Data Loading
# ===================================

def load_keypoints(keypoints_dir, split):
    """Load keypoints data for a specific split and convert to model features."""
    keypoints_file = os.path.join(keypoints_dir, f"{split}_keypoints.pkl")
    
    if not os.path.exists(keypoints_file):
        print(f"Keypoints file not found: {keypoints_file}")
        return None, None
    
    with open(keypoints_file, 'rb') as f:
        keypoints_data = pickle.load(f)
    
    features = []
    labels = []
    
    for sample in keypoints_data:
        keypoints = np.array(sample['keypoints'])
        class_name = sample['class']
        class_idx = CLASSES.index(class_name)
        
        # Flatten keypoints for model input
        feature_vector = keypoints_to_features(keypoints)
        
        features.append(feature_vector)
        labels.append(class_idx)
    
    return np.array(features), np.array(labels)

def create_dataloader(features, labels, batch_size=32, shuffle=True):
    """Create a PyTorch DataLoader for the keypoints data."""
    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )
    
    return dataloader

# ===================================
# Evaluation Functions
# ===================================

def calculate_pose_similarity(pose1, pose2):
    """Calculate similarity between two poses (0-1 scale)."""
    # Get confidence values
    conf1 = pose1[:, 2]
    conf2 = pose2[:, 2]
    
    # Consider only keypoints with good confidence in both poses
    good_keypoints = np.logical_and(conf1 > 0.5, conf2 > 0.5)
    
    if np.sum(good_keypoints) == 0:
        return 0.0
    
    # Calculate Euclidean distance between corresponding keypoints
    distances = np.sqrt(np.sum((pose1[good_keypoints, :2] - pose2[good_keypoints, :2])**2, axis=1))
    
    # Convert distances to similarities (exponential decay)
    similarities = np.exp(-distances)
    
    return np.mean(similarities)

def calculate_statistics(dataset_dir):
    """Calculate class distribution statistics."""
    stats = {"train": {}, "val": {}, "test": {}}
    
    for split in stats.keys():
        for cls in CLASSES:
            class_dir = os.path.join(dataset_dir, split, cls)
            if os.path.exists(class_dir):
                img_count = len(os.listdir(class_dir))
                stats[split][cls] = img_count
    
    return pd.DataFrame(stats)

def save_results_to_yaml(results, filepath):
    """Save results dictionary to YAML file."""
    with open(filepath, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)

def load_results_from_yaml(filepath):
    """Load results from YAML file."""
    with open(filepath, 'r') as f:
        results = yaml.safe_load(f)
    
    return results
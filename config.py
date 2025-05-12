"""
Configuration file for the Yoga Pose Checker project.
Contains paths, model parameters, and other settings.
"""
import os
import torch

# Directory structure 
DATA_DIR = "data"
LOGS_DIR = "logs"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Data directories
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
KEYPOINTS_DIR = os.path.join(DATA_DIR, "keypoints")
VISUALIZATIONS_DIR = os.path.join(DATA_DIR, "visualizations")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Dataset info
# 5 yoga poses from the Kaggle dataset
CLASSES = ["downdog", "goddess", "plank", "tree", "warrior2"]
NUM_CLASSES = len(CLASSES)

# Splitting parameters
VALIDATION_SPLIT = 0.2  # 20% validation split
RANDOM_SEED = 42

# Image preprocessing
IMAGE_SIZE = (256, 256)  # Resize images to this dimension
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization values
NORMALIZE_STD = [0.229, 0.224, 0.225]

# MediaPipe settings
MEDIAPIPE_DETECTION_CONFIDENCE = 0.5  # Min confidence for pose detection
MEDIAPIPE_TRACKING_CONFIDENCE = 0.5   # Min confidence for pose tracking

# Use GPU if available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# MediaPipe provides 33 keypoints, different from COCO's 17
# Indices for important landmarks in MediaPipe pose model
KEYPOINTS_INDEX = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32
}

# Key body parts for yoga pose analysis
ESSENTIAL_KEYPOINTS = [
    KEYPOINTS_INDEX['left_shoulder'],
    KEYPOINTS_INDEX['right_shoulder'],
    KEYPOINTS_INDEX['left_elbow'],
    KEYPOINTS_INDEX['right_elbow'],
    KEYPOINTS_INDEX['left_wrist'],
    KEYPOINTS_INDEX['right_wrist'],
    KEYPOINTS_INDEX['left_hip'],
    KEYPOINTS_INDEX['right_hip'],
    KEYPOINTS_INDEX['left_knee'],
    KEYPOINTS_INDEX['right_knee'],
    KEYPOINTS_INDEX['left_ankle'],
    KEYPOINTS_INDEX['right_ankle']
]

# Neural network parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Model architecture
INPUT_SIZE = 33 * 3  # 33 MediaPipe keypoints with (x, y, confidence)
HIDDEN_LAYERS = [256, 128, 64]
DROPOUT_RATE = 0.3

# Pose comparison thresholds
POSE_SIMILARITY_THRESHOLD = 0.75
JOINT_ANGLE_TOLERANCE = 15  # degrees
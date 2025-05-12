import os
import sys
import argparse
import subprocess
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import mediapipe as mp
from PIL import Image

# Import configuration
from config import (
    KEYPOINTS_DIR, MODELS_DIR, CLASSES, DEVICE,
    IMAGE_SIZE, DATA_DIR, PROCESSED_DIR, RESULTS_DIR,
    MEDIAPIPE_DETECTION_CONFIDENCE
)

# Import utility functions
from utils import extract_pose_angles, setup_processed_directories, setup_keypoints_directories

# Import model functions
from model import YogaPoseClassifier, check_pose_correctness, load_reference_poses

def preprocess_image(img_path, target_size=IMAGE_SIZE):
    """Preprocess a single image for pose extraction or inference."""
    try:
        # Read image
        img = Image.open(img_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        resampling = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
        img = img.resize(target_size, resampling)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        return img_array
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# ===================================
# Environment Check
# ===================================

def check_environment():
    """Check if the environment is properly set up for the project."""
    print("=" * 50)
    print("CHECKING ENVIRONMENT")
    print("=" * 50)
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Check for required directories
    print("\nChecking directories...")
    required_dirs = [DATA_DIR, MODELS_DIR, RESULTS_DIR]
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory} exists")
        else:
            print(f"✗ {directory} does not exist")
            os.makedirs(directory)
            print(f"  Created {directory}")
    
    # Check for original dataset
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print("\n✓ Dataset found")
        classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        print(f"  Classes: {', '.join(classes)}")
        train_images = sum([len([f for f in os.listdir(os.path.join(train_dir, cls)) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) for cls in classes])
        test_images = sum([len([f for f in os.listdir(os.path.join(test_dir, cls)) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) for cls in classes])
        print(f"  Training images: {train_images}")
        print(f"  Test images: {test_images}")
    else:
        print("\n✗ Dataset not found")
        print("  Please place your yoga pose dataset in the data directory")
        print("  Expected structure: data/train/class1, data/train/class2, etc.")
    
    # Check for processed data
    if os.path.exists(PROCESSED_DIR):
        print("\n✓ Processed data directory exists")
        has_train = os.path.exists(os.path.join(PROCESSED_DIR, "train"))
        has_val = os.path.exists(os.path.join(PROCESSED_DIR, "val"))
        has_test = os.path.exists(os.path.join(PROCESSED_DIR, "test"))
        print(f"  Contains: {'train' if has_train else ''} {'val' if has_val else ''} {'test' if has_test else ''}")
    else:
        print("\n✗ Processed data directory not found")
        print("  Run preprocessing step first")
    
    # Check for keypoints data
    if os.path.exists(KEYPOINTS_DIR):
        print("\n✓ Keypoints data directory exists")
        if os.path.exists(os.path.join(KEYPOINTS_DIR, "train_keypoints.pkl")):
            print("  Training keypoints found")
        if os.path.exists(os.path.join(KEYPOINTS_DIR, "val_keypoints.pkl")):
            print("  Validation keypoints found")
        if os.path.exists(os.path.join(KEYPOINTS_DIR, "test_keypoints.pkl")):
            print("  Test keypoints found")
    else:
        print("\n✗ Keypoints data directory not found")
        print("  Run pose extraction step first")
    
    # Check for trained model
    model_path = os.path.join(MODELS_DIR, "yoga_pose_classifier.pth")
    if os.path.exists(model_path):
        print("\n✓ Trained model found")
    else:
        print("\n✗ Trained model not found")
        print("  Run model training step first")
    
    # Check for MediaPipe
    try:
        import mediapipe as mp
        print("\n✓ MediaPipe is available")
    except ImportError:
        print("\n✗ MediaPipe is not available")
        print("  Install MediaPipe with 'pip install mediapipe'")
    
    print("\nEnvironment check complete!")

# ===================================
# Pipeline Steps
# ===================================

def run_preprocessing():
    """Run the data preprocessing step."""
    print("=" * 50)
    print("RUNNING PREPROCESSING")
    print("=" * 50)
    
    # Check if processed data already exists
    train_processed = os.path.exists(os.path.join(PROCESSED_DIR, "train"))
    val_processed = os.path.exists(os.path.join(PROCESSED_DIR, "val"))
    test_processed = os.path.exists(os.path.join(PROCESSED_DIR, "test"))
    
    if train_processed and val_processed and test_processed:
        print("\nProcessed data already exists. Skipping preprocessing.")
        return
    
    try:
        subprocess.run([sys.executable, "preprocessing.py"], check=True)
        print("\nPreprocessing completed successfully!")
    except subprocess.CalledProcessError:
        print("\nPreprocessing failed. Check the error message above.")
        sys.exit(1)

def run_pose_extraction():
    """Run the pose extraction step."""
    print("=" * 50)
    print("RUNNING POSE EXTRACTION")
    print("=" * 50)
    
    try:
        # Check if MediaPipe is available
        import mediapipe as mp
    except ImportError:
        print("Error: MediaPipe is not available. Please install it first.")
        print("Run: pip install mediapipe")
        sys.exit(1)
    
    try:
        subprocess.run([sys.executable, "extract_pose_model.py"], check=True)
        print("\nPose extraction completed successfully!")
    except subprocess.CalledProcessError:
        print("\nPose extraction failed. Check the error message above.")
        sys.exit(1)

def run_model_training():
    """Run the model training step."""
    print("=" * 50)
    print("RUNNING MODEL TRAINING")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "model.py"], check=True)
        print("\nModel training completed successfully!")
    except subprocess.CalledProcessError:
        print("\nModel training failed. Check the error message above.")
        sys.exit(1)

def run_evaluation():
    """Run the model evaluation step."""
    print("=" * 50)
    print("RUNNING MODEL EVALUATION")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "evaluate.py"], check=True)
        print("\nEvaluation completed successfully!")
    except subprocess.CalledProcessError:
        print("\nEvaluation failed. Check the error message above.")
        sys.exit(1)

def run_full_pipeline():
    """Run the complete pipeline from preprocessing to evaluation."""
    print("=" * 50)
    print("RUNNING FULL PIPELINE")
    print("=" * 50)
    
    start_time = time.time()
    
    # 1. Preprocessing
    print("\n[Step 1/4] Running preprocessing...")
    run_preprocessing()
    
    # 2. Pose extraction
    print("\n[Step 2/4] Running pose extraction...")
    run_pose_extraction()
    
    # 3. Model training
    print("\n[Step 3/4] Running model training...")
    run_model_training()
    
    # 4. Evaluation
    print("\n[Step 4/4] Running evaluation...")
    run_evaluation()
    
    end_time = time.time()
    print(f"\nFull pipeline completed in {end_time - start_time:.2f} seconds!")

# ===================================
# Demo Function
# ===================================

def process_single_image(image_path):
    """
    Process a single image and check the yoga pose.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with results or None on failure
    """
    try:
        # Load model
        model_path = os.path.join(MODELS_DIR, "yoga_pose_classifier.pth")
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found. Run training first.")
            return None
        
        model = YogaPoseClassifier().to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Load reference poses
        reference_poses = load_reference_poses()
        if not reference_poses:
            print("Warning: No reference poses found. Correctness checking will be limited.")
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found.")
            return None
        
        # Load and preprocess the image
        img = preprocess_image(image_path)
        if img is None:
            print(f"Error: Could not process image {image_path}")
            return None
        
        # Convert to RGB for MediaPipe
        img_rgb = (img * 255).astype(np.uint8)
        
        # Initialize MediaPipe pose detector
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE
        ) as pose:
            # Process the image with MediaPipe
            results = pose.process(img_rgb)
            
            if not results.pose_landmarks:
                print("Error: No pose detected in the image")
                return None
            
            # Extract keypoints
            h, w, _ = img_rgb.shape
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                x = landmark.x * w
                y = landmark.y * h
                visibility = landmark.visibility
                keypoints.append([x, y, visibility])
            
            keypoints = np.array(keypoints)
            
            # Create visualization
            vis_img = img_rgb.copy()
            mp_drawing.draw_landmarks(
                vis_img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Predict pose class
            features = keypoints.flatten()
            inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)[0]
                _, predicted = torch.max(outputs, 1)
                predicted_class = CLASSES[predicted.item()]
                confidence = probabilities[predicted.item()].item()
            
            # Check pose correctness
            pose_angles = extract_pose_angles(keypoints)
            
            if pose_angles and predicted_class in reference_poses and reference_poses[predicted_class]:
                is_correct, feedback = check_pose_correctness(pose_angles, reference_poses, predicted_class)
            else:
                is_correct = None
                feedback = "Unable to check pose correctness."
            
            # Prepare results
            class_probs = {cls: float(probabilities[i].item()) for i, cls in enumerate(CLASSES)}
            
            results = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': class_probs,
                'is_correct': is_correct,
                'feedback': feedback,
                'visualization': vis_img
            }
            
            return results
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def main():
    """Main function to run the Yoga Pose Checker project."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Yoga Pose Checker")
    
    # Add arguments for different pipeline steps
    parser.add_argument("--check", action="store_true", help="Check environment and data")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--extract", action="store_true", help="Run pose extraction")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--pipeline", action="store_true", help="Run the full pipeline")
    parser.add_argument("--image", type=str, help="Process a single image")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Run selected components
    if args.check:
        check_environment()
    
    if args.preprocess:
        run_preprocessing()
    
    if args.extract:
        run_pose_extraction()
    
    if args.train:
        run_model_training()
    
    if args.evaluate:
        run_evaluation()
    
    if args.pipeline:
        run_full_pipeline()
    
    if args.image:
        results = process_single_image(args.image)
        
        if results:
            # Display results
            print("\nResults:")
            print(f"Predicted Pose: {results['predicted_class']} (Confidence: {results['confidence']*100:.2f}%)")
            
            # Show top 3 probabilities
            print("\nTop Probabilities:")
            sorted_probs = sorted(results['class_probabilities'].items(), key=lambda x: x[1], reverse=True)
            for cls, prob in sorted_probs[:3]:
                print(f"  {cls}: {prob*100:.2f}%")
            
            # Show pose correctness
            if results['is_correct'] is not None:
                print("\nPose Correctness:")
                if results['is_correct']:
                    print("✓ Your pose is correct!")
                else:
                    print("✗ Your pose needs improvement")
                
                # Print feedback
                print("\nFeedback:")
                print(results['feedback'])
            
            # Save visualization
            output_path = os.path.join(RESULTS_DIR, "single_image_result.jpg")
            cv2.imwrite(output_path, results['visualization'])
            print(f"\nVisualization saved to {output_path}")
            
            # Display the visualization
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(results['visualization'], cv2.COLOR_BGR2RGB))
            plt.title(f"Predicted: {results['predicted_class']}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()
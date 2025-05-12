
import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import mediapipe as mp

# Import utility functions
from utils import (
    setup_keypoints_directories, 
    extract_pose_angles, 
    normalize_keypoints, 
    save_results_to_yaml
)

# Import configuration
from config import (
    PROCESSED_DIR, KEYPOINTS_DIR, VISUALIZATIONS_DIR, CLASSES,
    DEVICE, IMAGE_SIZE, MEDIAPIPE_DETECTION_CONFIDENCE
)

# Set up MediaPipe pose components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_keypoints(image_path, pose_detector):
    """
    Extract body keypoints from a single image.
    
    Args:
        image_path: Path to the image file
        pose_detector: MediaPipe pose detector
        
    Returns:
        Tuple of (keypoints array, visualization image) or (None, None) on failure
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        
        # Convert to RGB (MediaPipe requires RGB input)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = pose_detector.process(rgb_img)
        
        # Check if pose was detected
        if results.pose_landmarks is None:
            return None, None
        
        # Create visualization
        vis_img = img.copy()
        mp_drawing.draw_landmarks(
            vis_img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Convert MediaPipe landmarks to numpy array (x, y, confidence)
        h, w, _ = img.shape
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            x = landmark.x * w  # Convert from normalized [0,1] to pixel coordinates
            y = landmark.y * h
            visibility = landmark.visibility  # MediaPipe's confidence/visibility
            keypoints.append([x, y, visibility])
        
        keypoints = np.array(keypoints)
        
        return keypoints, vis_img
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def process_dataset():
    """
    Process the entire yoga dataset and extract keypoints from all images.
    
    Returns:
        Dictionary containing all extracted keypoints
    """
    # Dictionary to store keypoints by split and class
    all_keypoints = {split: {cls: [] for cls in CLASSES} for split in ["train", "val", "test"]}
    
    # Dictionary to store reference poses for each class
    reference_poses = {cls: [] for cls in CLASSES}
    
    # Initialize MediaPipe pose detector
    with mp_pose.Pose(
        static_image_mode=True,  # For still images
        model_complexity=2,  # 0=Lite, 1=Full, 2=Heavy
        min_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE
    ) as pose_detector:
        
        # Process each data split
        for split in ["train", "val", "test"]:
            print(f"\nProcessing {split} split...")
            
            for cls in CLASSES:
                print(f"  Processing {cls} class...")
                class_dir = os.path.join(PROCESSED_DIR, split, cls)
                
                # Skip if directory doesn't exist
                if not os.path.exists(class_dir):
                    print(f"  Warning: Directory {class_dir} does not exist. Skipping.")
                    continue
                
                # Get image files (exclude augmented images for reference poses)
                if split == "train":
                    images = [f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                             and not f.startswith(("flip_", "rot_"))]
                else:
                    images = [f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Process each image
                for img_name in tqdm(images, desc=f"{split}-{cls}"):
                    img_path = os.path.join(class_dir, img_name)
                    
                    # Extract keypoints
                    keypoints, vis_img = extract_keypoints(img_path, pose_detector)
                    
                    if keypoints is not None:
                        sample_id = os.path.splitext(img_name)[0]
                        
                        # Normalize keypoints
                        normalized_keypoints = normalize_keypoints(keypoints, IMAGE_SIZE)
                        
                        # Extract joint angles
                        pose_angles = extract_pose_angles(keypoints)
                        
                        # Store keypoint data
                        keypoints_data = {
                            'sample_id': sample_id,
                            'class': cls,
                            'keypoints': keypoints.tolist(),
                            'normalized_keypoints': normalized_keypoints.tolist(),
                            'pose_angles': pose_angles
                        }
                        
                        all_keypoints[split][cls].append(keypoints_data)
                        
                        # Add non-augmented training samples as reference poses
                        if split == "train" and not img_name.startswith(("flip_", "rot_")) and pose_angles:
                            reference_poses[cls].append({
                                'sample_id': sample_id,
                                'pose_angles': pose_angles
                            })
                        
                        # Save visualization
                        if vis_img is not None:
                            vis_dir = os.path.join(VISUALIZATIONS_DIR, split, cls)
                            os.makedirs(vis_dir, exist_ok=True)
                            vis_path = os.path.join(vis_dir, f"{sample_id}_pose.jpg")
                            cv2.imwrite(vis_path, vis_img)
    
    # Save keypoints
    save_keypoints(all_keypoints, reference_poses)
    
    return all_keypoints

def save_keypoints(all_keypoints, reference_poses):
    """Save extracted keypoints and reference poses to disk."""
    # Save keypoints by class for each split
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            if all_keypoints[split][cls]:
                output_dir = os.path.join(KEYPOINTS_DIR, split)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{cls}_keypoints.pkl")
                with open(output_file, 'wb') as f:
                    pickle.dump(all_keypoints[split][cls], f)
    
    # Save combined keypoints file for each split
    for split in ["train", "val", "test"]:
        combined_keypoints = []
        for cls in CLASSES:
            combined_keypoints.extend(all_keypoints[split][cls])
        
        if combined_keypoints:
            output_file = os.path.join(KEYPOINTS_DIR, f"{split}_keypoints.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(combined_keypoints, f)
    
    # Save reference poses for each class
    for cls in CLASSES:
        if reference_poses[cls]:
            output_file = os.path.join(KEYPOINTS_DIR, f"reference_{cls}_poses.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(reference_poses[cls], f)

def analyze_extraction_results(all_keypoints):
    """
    Calculate success rates for keypoint extraction.
    
    Args:
        all_keypoints: Dictionary of extracted keypoints
        
    Returns:
        Dictionary of success rates
    """
    success_rates = {}
    
    print("\nKeypoint Extraction Analysis:")
    
    for split in ["train", "val", "test"]:
        success_rates[split] = {}
        print(f"\n{split.capitalize()} Split:")
        
        total_images = 0
        total_successful = 0
        
        for cls in CLASSES:
            # Count total images
            class_dir = os.path.join(PROCESSED_DIR, split, cls)
            image_count = 0
            if os.path.exists(class_dir):
                image_count = len([f for f in os.listdir(class_dir) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            # Count successful extractions
            successful_count = len(all_keypoints[split][cls])
            
            # Calculate success rate
            success_rate = (successful_count / image_count) * 100 if image_count > 0 else 0
            success_rates[split][cls] = success_rate
            
            print(f"  {cls}: {successful_count}/{image_count} ({success_rate:.2f}%)")
            
            total_images += image_count
            total_successful += successful_count
        
        # Overall success rate for this split
        overall_rate = (total_successful / total_images) * 100 if total_images > 0 else 0
        success_rates[split]['overall'] = overall_rate
        print(f"  Overall: {total_successful}/{total_images} ({overall_rate:.2f}%)")
    
    return success_rates

def main():
    """Main function to run the keypoint extraction pipeline."""
    print("=" * 50)
    print("YOGA POSE KEYPOINT EXTRACTION")
    print("=" * 50)
    
    # Check if keypoints already exist
    if (os.path.exists(os.path.join(KEYPOINTS_DIR, "train_keypoints.pkl")) and
        os.path.exists(os.path.join(KEYPOINTS_DIR, "val_keypoints.pkl")) and
        os.path.exists(os.path.join(KEYPOINTS_DIR, "test_keypoints.pkl"))):
        print("\nKeypoints data already exists. Skipping extraction.")
        return
    
    # Create necessary directories
    print("\nSetting up directories...")
    setup_keypoints_directories()
    
    # Process dataset and extract keypoints
    print("\nExtracting keypoints from images...")
    all_keypoints = process_dataset()
    
    # Analyze results
    success_rates = analyze_extraction_results(all_keypoints)
    
    # Save analysis results
    analysis_results = {
        "success_rates": success_rates,
        "keypoints_extracted": True
    }
    save_results_to_yaml(analysis_results, os.path.join(KEYPOINTS_DIR, "extraction_results.yaml"))
    
    print(f"\nKeypoint extraction complete! Data ready for model training.")

if __name__ == "__main__":
    main()
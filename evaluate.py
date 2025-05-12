"""
Evaluation script for the Yoga Pose Checker project.
Assesses model performance and pose correctness.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# Import from model.py
from model import YogaPoseClassifier, check_pose_correctness, load_reference_poses

# Import from utils module
from utils import (
    load_keypoints, create_dataloader, extract_pose_angles,
    save_results_to_yaml
)

# Import configuration
from config import (
    KEYPOINTS_DIR, MODELS_DIR, RESULTS_DIR, CLASSES, NUM_CLASSES,
    DEVICE, BATCH_SIZE, JOINT_ANGLE_TOLERANCE
)

def evaluate_model(model, test_loader):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained classification model
        test_loader: DataLoader for test data
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    # Get predictions for all test samples
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating model"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    report = classification_report(all_labels, all_predictions, 
                                  target_names=CLASSES, output_dict=True)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    return metrics, all_predictions, all_labels

def plot_confusion_matrix(confusion_matrix, save_path):
    """
    Plot and save confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
               xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_pose_correctness(test_keypoints_data, reference_poses):
    """
    Analyze pose correctness across the test set.
    
    Args:
        test_keypoints_data: Test keypoint data
        reference_poses: Reference poses by class
        
    Returns:
        Dictionary with correctness analysis results
    """
    # Initialize correctness counts
    correctness = {
        'total': len(test_keypoints_data),
        'correct_poses': 0,
        'by_class': {cls: {'total': 0, 'correct': 0} for cls in CLASSES},
        'common_issues': {}
    }
    
    # Check each test sample
    for sample in tqdm(test_keypoints_data, desc="Analyzing pose correctness"):
        true_class = sample['class']
        correctness['by_class'][true_class]['total'] += 1
        
        # Extract keypoints and angles
        keypoints = np.array(sample['keypoints'])
        pose_angles = extract_pose_angles(keypoints)
        
        if pose_angles and true_class in reference_poses and reference_poses[true_class]:
            # Check against reference poses
            is_correct, feedback = check_pose_correctness(pose_angles, reference_poses, true_class)
            
            if is_correct:
                correctness['correct_poses'] += 1
                correctness['by_class'][true_class]['correct'] += 1
            
            # Track issues by looking at feedback
            if not is_correct and feedback:
                lines = feedback.split('\n')
                for line in lines[1:]:  # Skip first line (summary)
                    if line.startswith('- Your'):
                        # Extract joint name
                        parts = line.split('angle')
                        if len(parts) > 0:
                            joint = parts[0].replace('- Your ', '').strip()
                            
                            if joint not in correctness['common_issues']:
                                correctness['common_issues'][joint] = 0
                            correctness['common_issues'][joint] += 1
    
    # Calculate percentages
    correctness['percent_correct'] = (correctness['correct_poses'] / correctness['total']) * 100
    
    for cls in CLASSES:
        cls_total = correctness['by_class'][cls]['total']
        if cls_total > 0:
            correctness['by_class'][cls]['percent_correct'] = (
                correctness['by_class'][cls]['correct'] / cls_total) * 100
    
    # Sort common issues by frequency
    correctness['common_issues'] = dict(
        sorted(correctness['common_issues'].items(), 
              key=lambda x: x[1], reverse=True)
    )
    
    return correctness

def plot_correctness_by_class(correctness_data, save_path):
    """
    Create a bar chart showing pose correctness by class.
    
    Args:
        correctness_data: Correctness analysis data
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    
    # Extract correctness percentages by class
    percentages = [correctness_data['by_class'][cls]['percent_correct'] 
                  if 'percent_correct' in correctness_data['by_class'][cls] else 0 
                  for cls in CLASSES]
    
    # Create bar chart
    plt.bar(CLASSES, percentages, color='lightgreen')
    plt.title('Pose Correctness by Yoga Pose Class')
    plt.xlabel('Yoga Pose Class')
    plt.ylabel('Correct Poses (%)')
    plt.ylim(0, 100)
    
    # Add value labels
    for i, v in enumerate(percentages):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_evaluation(model_path=None):
    """
    Run evaluation on the trained model.
    
    Args:
        model_path: Path to the model file (optional)
        
    Returns:
        Dictionary of evaluation results
    """
    print("=" * 50)
    print("YOGA POSE CLASSIFIER EVALUATION")
    print("=" * 50)
    
    # Load model
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "yoga_pose_classifier.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return None
    
    model = YogaPoseClassifier().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
    
    # Load test data
    print("\nLoading test data...")
    test_features, test_labels = load_keypoints(KEYPOINTS_DIR, 'test')
    
    if test_features is None:
        print("Error: Could not load test data.")
        return None
    
    print(f"Test samples: {len(test_features)}")
    
    # Create test data loader
    test_loader = create_dataloader(test_features, test_labels, batch_size=BATCH_SIZE, shuffle=False)
    
    # Evaluate classification performance
    print("\nEvaluating classification performance...")
    metrics, predictions, true_labels = evaluate_model(model, test_loader)
    
    # Print summary metrics
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=CLASSES))
    
    # Create confusion matrix visualization
    print("\nGenerating confusion matrix...")
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    
    # Load reference poses and test keypoints for correctness analysis
    reference_poses = load_reference_poses()
    
    test_keypoints_file = os.path.join(KEYPOINTS_DIR, "test_keypoints.pkl")
    with open(test_keypoints_file, 'rb') as f:
        test_keypoints_data = pickle.load(f)
    
    # Analyze pose correctness
    print("\nAnalyzing pose correctness...")
    correctness_analysis = analyze_pose_correctness(test_keypoints_data, reference_poses)
    
    # Print correctness summary
    print(f"\nPose correctness rate: {correctness_analysis['correct_poses']}/{correctness_analysis['total']} "
          f"({correctness_analysis['percent_correct']:.2f}%)")
    
    print("\nPose correctness by class:")
    for cls in CLASSES:
        cls_data = correctness_analysis['by_class'][cls]
        if cls_data['total'] > 0:
            print(f"  {cls}: {cls_data['correct']}/{cls_data['total']} "
                  f"({cls_data['percent_correct']:.2f}%)")
    
    print("\nMost common pose issues:")
    for joint, count in list(correctness_analysis['common_issues'].items())[:3]:
        print(f"  {joint}: {count} occurrences")
    
    # Create correctness visualization
    print("\nGenerating correctness visualization...")
    plot_correctness_by_class(correctness_analysis, 
                             os.path.join(RESULTS_DIR, "pose_correctness.png"))
    
    # Combine results
    evaluation_results = {
        'classification_metrics': metrics,
        'correctness_analysis': correctness_analysis
    }
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.yaml")
    save_results_to_yaml(evaluation_results, results_path)
    print(f"\nEvaluation results saved to {results_path}")
    
    return evaluation_results

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate the yoga pose classification model")
    parser.add_argument("--model", type=str, help="Path to the model file")
    args = parser.parse_args()
    
    # Run evaluation
    run_evaluation(model_path=args.model)
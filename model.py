import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Import from utils module
from utils import load_keypoints, create_dataloader, extract_pose_angles, save_results_to_yaml

# Import configuration
from config import (
    KEYPOINTS_DIR, MODELS_DIR, RESULTS_DIR, CLASSES, NUM_CLASSES,
    DEVICE, INPUT_SIZE, HIDDEN_LAYERS, DROPOUT_RATE,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
    JOINT_ANGLE_TOLERANCE, POSE_SIMILARITY_THRESHOLD
)

# Ensure directories exist
for directory in [MODELS_DIR, RESULTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# ===================================
# Model Architecture
# ===================================

class YogaPoseClassifier(nn.Module):
    """
    Neural network model for yoga pose classification.
    Takes keypoints as input and outputs pose class probabilities.
    """
    def __init__(self, input_size=INPUT_SIZE, hidden_layers=HIDDEN_LAYERS, 
                 num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super(YogaPoseClassifier, self).__init__()
        
        # Create model layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], num_classes))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)

# ===================================
# Training Function
# ===================================

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE):
    """
    Train the yoga pose classification model with early stopping.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs: Maximum training epochs
        patience: Early stopping patience
        
    Returns:
        Dictionary with training history
    """
    # Track metrics during training
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Early stopping setup
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history

def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig(os.path.join(RESULTS_DIR, "training_history.png"))
    
    plt.close()

# ===================================
# Pose Correctness Functions
# ===================================

def load_reference_poses():
    """
    Load reference poses for each class.
    
    Returns:
        Dictionary: Reference poses by class
    """
    reference_poses = {}
    
    for cls in CLASSES:
        ref_file = os.path.join(KEYPOINTS_DIR, f"reference_{cls}_poses.pkl")
        if os.path.exists(ref_file):
            with open(ref_file, 'rb') as f:
                reference_poses[cls] = pickle.load(f)
    
    return reference_poses

def check_pose_correctness(pose_angles, reference_poses, class_name, tolerance=JOINT_ANGLE_TOLERANCE):
    """
    Check if a pose is correct by comparing joint angles with reference poses.
    
    Args:
        pose_angles: Joint angles of the pose
        reference_poses: Reference poses by class
        class_name: Class name
        tolerance: Angle tolerance in degrees
        
    Returns:
        tuple: (is_correct, feedback)
    """
    # Handle missing data
    if not pose_angles or class_name not in reference_poses or not reference_poses[class_name]:
        return False, "Unable to check pose correctness due to missing data."
    
    # Get reference angles for the class
    ref_angles_list = [ref['pose_angles'] for ref in reference_poses[class_name] if ref['pose_angles']]
    
    if not ref_angles_list:
        return False, "No reference poses available for comparison."
    
    # Compare with each reference pose and find the best match
    best_match_score = 0
    best_match_idx = 0
    
    for i, ref_angles in enumerate(ref_angles_list):
        match_score = 0
        total_joints = 0
        
        for joint, angle in pose_angles.items():
            if joint in ref_angles:
                total_joints += 1
                if abs(angle - ref_angles[joint]) <= tolerance:
                    match_score += 1
        
        # Calculate match percentage
        if total_joints > 0:
            match_percentage = match_score / total_joints
            if match_percentage > best_match_score:
                best_match_score = match_percentage
                best_match_idx = i
    
    # Generate feedback
    is_correct = best_match_score >= POSE_SIMILARITY_THRESHOLD
    
    feedback = []
    if is_correct:
        feedback.append(f"Good job! Your {class_name} pose is {best_match_score:.1%} correct.")
    else:
        feedback.append(f"Your {class_name} pose needs improvement ({best_match_score:.1%} accuracy).")
    
    # Add detailed joint feedback
    best_ref = ref_angles_list[best_match_idx]
    for joint, angle in pose_angles.items():
        if joint in best_ref:
            diff = angle - best_ref[joint]
            if abs(diff) > tolerance:
                if diff > 0:
                    feedback.append(f"- Your {joint.replace('_', ' ')} angle is too wide by {abs(diff):.1f} degrees")
                else:
                    feedback.append(f"- Your {joint.replace('_', ' ')} angle is too narrow by {abs(diff):.1f} degrees")
    
    return is_correct, "\n".join(feedback)

# ===================================
# Main Function
# ===================================

def main():
    """Train and evaluate the yoga pose classification model."""
    print("=" * 50)
    print("YOGA POSE CLASSIFICATION MODEL")
    print("=" * 50)
    
    # Check if model exists already
    model_path = os.path.join(MODELS_DIR, "yoga_pose_classifier.pth")
    if os.path.exists(model_path):
        print(f"\nFound existing model at {model_path}")
        response = input("Do you want to retrain the model? (y/n): ")
        
        if response.lower() != 'y':
            print("Skipping training. Use evaluate.py to test the model.")
            return
    
    # Load data
    print("\nLoading training and validation data...")
    train_features, train_labels = load_keypoints(KEYPOINTS_DIR, 'train')
    val_features, val_labels = load_keypoints(KEYPOINTS_DIR, 'val')
    
    if train_features is None or val_features is None:
        print("Error: Could not load training or validation data.")
        return
    
    print(f"Training samples: {len(train_features)}")
    print(f"Validation samples: {len(val_features)}")
    
    # Create data loaders
    train_loader = create_dataloader(train_features, train_labels, batch_size=BATCH_SIZE)
    val_loader = create_dataloader(val_features, val_labels, batch_size=BATCH_SIZE)
    
    # Initialize model
    print("\nInitializing model...")
    model = YogaPoseClassifier().to(DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("\nTraining model...")
    start_time = time.time()
    history = train_model(model, train_loader, val_loader, criterion, optimizer)
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds.")
    
    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save training history
    history_path = os.path.join(RESULTS_DIR, "training_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    # Plot training history
    plot_training_history(history)
    
    # Quick test on validation data
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"\nValidation accuracy: {100 * correct / total:.2f}%")
    print("\nModel training complete! Use evaluate.py for detailed evaluation.")

if __name__ == "__main__":
    main()
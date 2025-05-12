# Yoga Pose Checker

## Project Overview
This application uses computer vision and machine learning to classify yoga poses and check their correctness. As a beginner to yoga, it's often difficult to verify if you're performing poses correctly without a personal trainer. This tool helps bridge that gap by analyzing body posture using pose estimation and comparing it against reference poses.

## Features
- **Pose Classification**: Identifies five different yoga poses (downdog, goddess, plank, tree, warrior2)
- **Posture Correctness**: Verifies if your pose is being performed correctly by comparing joint angles
- **Detailed Feedback**: Provides specific feedback on which joints need adjustment
- **Visual Representation**: Shows pose skeleton visualization

## Dataset
The dataset used in this project is the Yoga Poses Dataset by Niharika Pandit, publicly available on Kaggle (https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset?resource=download). It contains labeled images of five common yoga poses:

- Downward Dog
- Goddess
- Plank
- Tree
- Warrior2

Each pose class includes varied examples collected from the web using Bing's API, and images may contain noise (e.g., watermarks or varying backgrounds). The dataset was split into training (70%), validation (15%), and test (15%) sets. Data augmentation techniques (flipping, rotation, scaling) were applied to improve generalization.

License Note: The dataset is provided under an open data license.

## Technical Implementation

### Pipeline
1. **Preprocessing**: Images are resized, normalized, and augmented to create a robust training set
2. **Pose Extraction**: MediaPipe is used to extract skeletal keypoints from yoga pose images
3. **Model Training**: A neural network classifies poses based on extracted keypoint features
4. **Evaluation**: Model performance is assessed with classification metrics and pose correctness verification

### Technologies Used
- **Python 3.10**: Primary programming language
- **PyTorch**: Deep learning framework for neural network implementation
- **MediaPipe**: For efficient and accurate pose estimation
- **NumPy/Pandas**: For data processing and analysis
- **Matplotlib/Seaborn**: For visualizations and result analysis

## Results
- **Classification Accuracy**: 95.00%
- **Pose Correctness**: The system can identify correct pose execution with detailed joint-level feedback
- **Best Performing Classes**: Warrior2 (95.33% classification, 81.31% correctness) and Plank (98.25% classification, 32.46% correctness)

## How to Run

### Prerequisites
- Python 3.8+ 
- Dependencies listed in requirements.txt

### Installation
1. Clone this repository
2. Set up a virtual environment: `python -m venv dne`
3. Activate the environment: 
   - Windows: `dne\Scripts\activate`
   - Linux/Mac: `source dne/bin/activate`
4. Install requirements: `pip install -r requirements.txt`

### Running the Application
- Full pipeline: `python main.py --pipeline`
- Individual components:
  - Preprocessing: `python main.py --preprocess`
  - Pose extraction: `python main.py --extract`
  - Model training: `python main.py --train`
  - Evaluation: `python main.py --evaluate`
  - Check a single image: `python main.py --image /path/to/image.jpg`

## Acknowledgments
This project was developed as part of the Deep Neural Engineering course in the AI master's program. It builds upon research in pose estimation and computer vision for human activity recognition.

## Author 
Jishnu Harinandansingh# dne_assignment2

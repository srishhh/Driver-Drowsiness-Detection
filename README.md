# Driver Drowsiness Detection System

## Project Overview

The Driver Drowsiness Detection System is a computer vision and deep learning project developed to explore the detection of potential driver drowsiness through eye-state classification.

The project uses a Convolutional Neural Network (CNN) to classify eye images as open or closed. It combines image preprocessing, facial feature detection, and deep learning with a live webcam-based detection script to demonstrate a potential approach to identifying prolonged eye closure.

The project explores the application of artificial intelligence and computer vision techniques to a road-safety problem.

## Objectives

- Develop a CNN model for classifying eye images as open or closed.
- Apply image preprocessing techniques to prepare eye images for model training.
- Explore facial and eye detection using OpenCV Haar Cascade classifiers.
- Integrate the trained model into a webcam-based detection pipeline.
- Implement a frame-based scoring mechanism and optional audio alert for potential prolonged eye closure.

## Technologies Used

- **Python:** Core programming language.
- **TensorFlow and Keras:** CNN development, training, and inference.
- **OpenCV:** Image processing, webcam access, and Haar Cascade-based face and eye detection.
- **NumPy:** Numerical operations and image array manipulation.
- **scikit-learn:** Dataset splitting and model development utilities.
- **Pillow (PIL):** Image handling and processing.
- **face_recognition:** Facial landmark detection used in the training notebook's eye-region extraction code.
- **Pygame:** Audio playback for the live-detection alert.
- **Jupyter Notebook:** Interactive development and experimentation.

## Project Structure

```text
Driver-Drowsiness-Detection/
|
|-- drivers_drowsiness_detection_system.ipynb
|-- live_detection.py
|-- best_model.h5
|-- alarm.wav
|-- dataset.zip
|-- dataset_new/
|   |-- train/
|       |-- Open/
|       |-- Closed/
|-- README.md
```

The structure above describes the expected project files and directories. The trained model, alarm audio file, and extracted dataset must be present where required by the corresponding code. Optional files may be absent if they have not been generated or supplied.

## System Workflow

The project consists of two main components.

### 1. CNN Model Training

The Jupyter Notebook contains the model-development pipeline:

1. Load eye images from the dataset.
2. Resize images to 80 × 80 pixels.
3. Normalize pixel values.
4. Prepare the image arrays and corresponding class labels.
5. Split the data into training and evaluation subsets.
6. Build a CNN using convolutional, max-pooling, fully connected, and dropout layers.
7. Train the model using the Adam optimizer and binary cross-entropy loss.
8. Evaluate the model and visualize training and validation metrics.
9. Save the trained model as `best_model.h5`.

### 2. Live Drowsiness Detection

The `live_detection.py` script integrates the trained CNN with a webcam-based detection pipeline.

The process involves:

1. Initialize the webcam and load the trained CNN.
2. Detect the driver's face using an OpenCV Haar Cascade classifier.
3. Detect eye regions within the face.
4. Resize and normalize detected eye images to match the CNN's expected input format.
5. Predict whether the detected eyes are open or closed.
6. Update a frame-based score when closed-eye predictions occur.
7. Display the detection status and trigger an optional audio alert when the configured threshold is reached.

The live-detection implementation uses a threshold-based heuristic to identify repeated closed-eye predictions. This is an experimental approach rather than a clinically validated measure of driver drowsiness.

## CNN Model Architecture

The model is implemented using TensorFlow/Keras and follows a Sequential CNN architecture.

The principal components include:

- Convolutional layers with ReLU activation for extracting visual features.
- Max-pooling layers for reducing spatial dimensions.
- A Flatten layer for converting feature maps into a one-dimensional representation.
- Fully connected Dense layers containing 256, 128, and 64 units.
- Dropout layers with a rate of 0.3 to help reduce overfitting.
- A sigmoid output layer for binary classification.

### Training Configuration

| Parameter | Value |
|---|---|
| Input image dimensions | 80 × 80 pixels |
| Input channels | 3 |
| Classification task | Binary classification |
| Loss function | Binary cross-entropy |
| Optimizer | Adam |
| Evaluation metric | Precision-Recall AUC |
| Batch size | 800 |
| Training epochs | 50 |
| Saved model filename | `best_model.h5` |

## Requirements

The project uses the following Python libraries:

- TensorFlow
- OpenCV
- NumPy
- scikit-learn
- Pillow
- face_recognition
- Pygame
- Jupyter Notebook

The `face_recognition` package may require additional system dependencies, including a compatible installation of `dlib`. Dependency versions may need adjustment depending on the Python environment.

## Evaluation

The training notebook includes model evaluation and plots of training and validation metrics. The live-detection script provides visual status feedback and a threshold-based closed-eye score.

Numerical performance results should be reported only after confirming the corresponding evaluation output. Results documented in the original project report may not necessarily reflect the performance of the current repository version.

## Limitations

- The CNN classifies eye images as open or closed; this classification alone does not establish that a driver is drowsy.
- Haar Cascade detection may be affected by lighting, glasses, head position, and image quality.
- The live-detection score is based on video frames rather than elapsed eye-closure time.
- Webcam detection may produce false positives or false negatives.
- The model's performance depends on the quality and diversity of its training data.
- The project has not been established as a validated or production-ready driver-safety system.

## Potential Future Improvements

- Introduce separate training, validation, and test datasets.
- Evaluate performance using precision, recall, F1-score, and a confusion matrix.
- Improve eye-region detection under different lighting conditions and head positions.
- Use elapsed eye-closure duration and temporal analysis to improve detection.
- Document dependency versions and provide a reproducible environment configuration.
- Evaluate the complete pipeline on representative video recordings.

## Conclusion

This project explores the application of computer vision and deep learning to driver drowsiness detection. It combines CNN-based eye-state classification with facial feature detection and a webcam-based inference pipeline.

The implementation provides practical exposure to image preprocessing, CNN architecture design, binary classification, model evaluation, and integration of a trained model into a computer vision application.

**Project category:** Computer Vision and Deep Learning  
**Primary task:** Eye-state classification  
**Framework:** TensorFlow/Keras  
**Programming language:** Python  
**Development environment:** Jupyter Notebook

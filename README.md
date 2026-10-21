# Driver Drowsiness Detection System

A computer vision and deep learning project that explores driver drowsiness detection through eye-state classification. The system uses facial landmark detection to extract eye regions and a Convolutional Neural Network (CNN) to classify images of eyes as **open or closed**.

This project was developed to explore the application of image processing, computer vision, and deep learning to a road-safety problem.

## Project Overview

Driver fatigue and drowsiness can affect attention and reaction time while driving. This project investigates one approach to detecting a potential indicator of drowsiness by analysing eye images.

The implementation focuses on preprocessing eye images and training a CNN to distinguish between open and closed eyes.

## Key Features

- **Eye Region Extraction:** Uses facial landmarks to identify and crop the left and right eye regions from images.
- **Image Preprocessing:** Resizes images to 80 × 80 pixels and normalizes pixel values.
- **Binary Classification:** Labels eye images as open or closed.
- **CNN Architecture:** Uses convolutional and pooling layers to learn visual features.
- **Model Training and Evaluation:** Trains the model using training and validation data and evaluates it on the held-out test split.
- **Model Saving:** Saves the trained model as `best_model.h5`.

## Technologies Used

- **Python** — Core programming language.
- **TensorFlow and Keras** — CNN development and model training.
- **OpenCV** — Image loading, resizing, and preprocessing.
- **face_recognition** — Facial landmark detection for extracting eye regions.
- **NumPy** — Numerical operations and array manipulation.
- **scikit-learn** — Splitting the dataset into training and test sets.
- **Pillow (PIL)** — Image handling and processing.
- **Jupyter Notebook** — Interactive development and experimentation.

## Model Architecture

The project uses a Sequential CNN built with TensorFlow/Keras.

The architecture includes:

1. Multiple convolutional layers with ReLU activation to learn image features.
2. Max-pooling layers to reduce spatial dimensions.
3. A Flatten layer to convert extracted features into a one-dimensional representation.
4. Fully connected Dense layers with 256, 128, and 64 units.
5. Dropout layers with a rate of 0.3 to help reduce overfitting.
6. A sigmoid output layer for binary classification.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Input image size | 80 × 80 pixels |
| Input channels | 3 (RGB/BGR image array) |
| Output | Binary classification |
| Loss function | Binary cross-entropy |
| Optimizer | Adam |
| Evaluation metric | Precision-Recall AUC |
| Batch size | 800 |
| Training epochs | 50 |

## Repository Contents

| File or directory | Description |
|---|---|
| `drivers_drowsiness_detection_system.ipynb` | Main notebook containing eye extraction, image preprocessing, CNN training, evaluation, and model saving. |
| `dataset.zip` | Compressed dataset archive expected by the notebook. |
| `dataset_new/train/Open/` | Directory containing open-eye training images. |
| `dataset_new/train/Closed/` | Directory containing closed-eye training images. |
| `best_model.h5` | Trained model file generated when the notebook reaches the model-saving step. |

*The dataset directories and model file are generated or required during execution; they may not be present in the repository initially.*

## Evaluation

The notebook evaluates the trained model on a held-out test split and uses **Precision-Recall AUC** as its evaluation metric.

Numerical evaluation results have not been documented in this README. Actual performance should be reported after running the notebook and recording the resulting metrics.

## Scope and Limitations

- The current implementation focuses on classifying eye images as open or closed.
- The notebook does not implement a complete real-time monitoring pipeline with webcam input or an audible alert.
- Closed-eye classification alone does not establish that a driver is drowsy.
- Model performance depends on dataset quality, image diversity, and the training configuration.
- The system is an experimental learning project and should not be used as a standalone driver-safety system.

## Potential Improvements

- Integrate real-time webcam-based eye detection.
- Track eye closure over time to explore indicators such as prolonged eye closure.
- Add an alert mechanism for potential drowsiness.
- Evaluate the model using additional metrics, including precision, recall, F1-score, and a confusion matrix.
- Improve reproducibility by documenting dataset sources, dependency versions, and experimental results.

## Project Summary

This project explores how computer vision and deep learning can be applied to an introductory driver-drowsiness detection problem. It provided practical experience with image preprocessing, facial landmarks, CNN architecture design, binary classification, and model evaluation.

**Project type:** Computer Vision and Deep Learning  
**Primary task:** Open-eye versus closed-eye classification  
**Framework:** TensorFlow/Keras  
**Development environment:** Jupyter Notebook

# Cats vs Dogs Image Classification

This repository contains code for classifying images of cats and dogs using a Support Vector Machine (SVM) classifier. The features for the classification are extracted using a pre-trained VGG16 model.

## Dataset

The dataset used for this project is the [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) from Kaggle.

## Project Structure

- `train/`: Directory containing the training images (cats and dogs).
- `test1/`: Directory containing the test images.
- `sampleSubmission.csv`: Sample submission file from Kaggle.
- `train_features.pkl`: Pickle file containing extracted features for training data.
- `test_features.pkl`: Pickle file containing extracted features for test data.
- `submission.csv`: Output file containing predictions for the test data.
- `svm_classifier.pkl`: Pickle file containing the trained SVM classifier.

## Requirements

- `Python 3.x`
- `NumPy`
- `pandas`
- `scikit-learn`
- `OpenCV`
- `Keras`
- `TensorFlow`

Install the required libraries using:

```bash
pip install numpy pandas scikit-learn opencv-python keras tensorflow

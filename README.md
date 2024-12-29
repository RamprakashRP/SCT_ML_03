# Cats vs Dogs Image Classification
![Cats vs Dogs Image Classification](https://github.com/BottomsNode/SCT_ML_3/blob/main/Task%20ML%203.png)

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
```

## Usage
  - `/`Training and Predicting
  - `/`Extract Features: Extract features from the images using the VGG16 model.
  - `/`Train SVM Classifier: Train the SVM classifier using the extracted features.
  - `/`Predict: Predict the labels for the test images.

## Predicting a Single Image
  To classify a single image, use the predict_image function included in the code.
  This function takes an image path as input and returns whether the image is of a cat or a dog.

## Acknowledgements
  This project uses the Dogs vs. Cats dataset from Kaggle. Special thanks to the Kaggle community for providing such a valuable dataset.

## License  
  This project is licensed under the MIT License.
  
### How to Use
  - Save this content in a file named `README.md` in the root directory of your project.

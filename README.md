# StyleSync

## Overview

StyleSync is a machine learning-based fashion recommendation system that leverages a Convolutional Neural Network (CNN) to classify clothing items and suggest outfits based on complementary color matching and brand preferences. The system is built using the VGG architecture and trained on the Fashion-MNIST dataset to enhance outfit recommendations through deep learning.

## Features

### Image Classification:
Uses a CNN model (VGG-based) to categorize clothing items.

### Fashion Recommendations: 
Provides outfit suggestions based on complementary color schemes and brand similarities.

### Dataset:
Trained on Fashion-MNIST, a dataset containing grayscale images of fashion items. This data set can be found here:

https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full

### Model Performance Optimization: 
Enhances accuracy through fine-tuning and hyperparameter tuning.

### Integration:
Can be incorporated into the StyleSync research project to analyze AI-assisted fashion recommendations.

## Installation

### Prerequisites

Ensure you have the following dependencies installed:

Python 3.8+

TensorFlow/Keras

NumPy

Matplotlib

OpenCV (for image preprocessing)

Scikit-learn

The data set: https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full

### Setup

Clone the repository:

git clone https://github.com/your-repo/stylesync.git
cd stylesync

Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

Install required dependencies:

pip install -r requirements.txt

Unpack data set into folder

## Usage

### Training the Model

Run the following command to train the CNN model on Fashion-MNIST:

python train.py

This will preprocess the dataset, train the model, and save it as model.h5.

### Making Predictions

To classify a clothing image, use:

python predict.py --image path/to/image.jpg

This will output the predicted category and possible outfit recommendations.

## Model Architecture

The CNN model is based on the VGG architecture with the following layers:

- Convolutional layers with ReLU activation

- Max pooling layers

- Fully connected layers

- Softmax output layer

## Future Improvements

### Enhance Dataset:
Expand beyond Fashion-MNIST with real-world clothing images.

### Style Customization:
Enable users to select outfit preferences (casual, formal, etc.).

### Brand Recognition:
Implement a model to detect and recommend based on brands.

### Mobile App Integration:
Develop a frontend interface for users to upload images and receive recommendations.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with improvements.

## License

This project is licensed under the MIT License.

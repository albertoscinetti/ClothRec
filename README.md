# Clothing Recognition Software in Law Enforcement Environments
This project uses deep learning techniques to classify images of clothing items, which can assist law enforcement in various scenarios. The software is built using Keras (via the TensorFlow library) and trained on the FashionMNIST dataset, making it a powerful tool for recognizing various clothing categories.

## Project Overview
### FashionMNIST dataset
The FashionMNIST dataset, provided by Zalando Research (2017), consists of 70,000 images of fashion products, split into:
- 60,000 training images
- 10,000 test images
Each image is already categorized into one of the 10 clothing categories, making this dataset ideal for training a clothing classification model.

### Clothing Categories
The 10 clothing categories in the FashionMNIST dataset are:
0: Shirt/top
1: Trouser
2: Pullover
3: Dresses
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot

### Model Architecture
We implement a Convolutional Neural Network (CNN) using the Sequential model from Keras to classify the clothing images. Here's a breakdown of the layers:

- Input Layer:
We start by flattening the input images (28x28 grayscale) to convert them into a 1D array.
- Dense Layer:
A fully connected layer with 128 neurons activated using the ReLU function.
- Output Layer:
The output layer has 10 neurons (one for each category) and is activated by the Softmax function to output probabilities.

### Model Compilation 
The model is compiled using the following settings:
- Optimizer: Adam (a variant of Stochastic Gradient Descent (SGD))
- Loss Function: sparse_categorical_crossentropy (used for multiclass classification)
- Metrics: Accuracy

### Training the Model 
We train the model with the following configurations:

- Batch Size: 32 (number of samples per gradient update)
- Epochs: 5 (number of passes through the entire training set)
- Multiple epochs ensure the model learns effectively by seeing the training data several times.

### Key Considerations
- Grayscale Images: All images in FashionMNIST are in grayscale, simplifying the model's architecture. However, to ensure consistency, we will process any input images for prediction in grayscale as well.

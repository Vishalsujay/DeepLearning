# Neural Network Models in PyTorch

This repository includes various neural network models implemented using PyTorch. Each model is organized into its own subdirectory for clarity and modularity. The repository is designed for beginners and advanced users interested in exploring and experimenting with neural networks.

## Models

This repository includes a range of models, each implemented in a dedicated subdirectory. Below are examples of what you can expect to see:

- **Linear Regression**: A simple linear regression model implemented using PyTorch.
- **Multiclassification**: A neural network model for multiclass classification tasks using the Iris dataset.
- **CNN (Convolutional Neural Network)**: A neural network architecture used for image processing tasks.
- **RNN (Recurrent Neural Network)**: A type of neural network suitable for sequence data and time-series problems.
- **Autoencoder**: A neural network used for unsupervised learning and dimensionality reduction.

### Current Models

#### 1. **Linear Regression**
- **Description**: A basic linear regression model demonstrating supervised learning.
- **Directory**: [linear_regression_pytorch](linear_regression_pytorch)
- **Dependencies**: PyTorch
- **How to Run**: 
  - Navigate to the `linear_regression_pytorch` directory.
  - Run the Python script: `python linear_regression.py`.

#### 2. **Multiclassification**
- **Description**: A neural network model for multiclass classification using the Iris dataset.
- **Directory**: [multiclassification_pytorch](multiclassification_pytorch)
- **Dependencies**: PyTorch, scikit-learn
- **How to Run**:
  - Navigate to the `multiclassification_pytorch` directory.
  - Run the Python script: `python multiclassification.py`.
- **Features**:
  - Two architectures: `Multiclassification` and `NeuralNetAdvance`.
  - Implements dropout for regularization.
  - Demonstrates the use of the Iris dataset for classification tasks.

#### 3. **Convolutional Neural Network (CNN)** (Coming Soon)
- **Description**: A model designed to process grid-like data, such as images, by using convolutional layers.
- **Directory**: [cnn_model_pytorch](cnn_model_pytorch)

#### 4. **Recurrent Neural Network (RNN)** (Coming Soon)
- **Description**: A type of neural network used for sequential data, such as time-series or natural language.
- **Directory**: [rnn_model_pytorch](rnn_model_pytorch)

#### 5. **Autoencoder** (Coming Soon)
- **Description**: An unsupervised learning model used for dimensionality reduction and feature extraction.
- **Directory**: [autoencoder_model_pytorch](autoencoder_model_pytorch)

## Installation

### Prerequisites

Ensure you have the following installed on your machine:

- **Python** 3.x
- **PyTorch** (visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for installation instructions)
- **scikit-learn** (for preprocessing data in the Multiclassification model)

You can install the necessary dependencies using `pip`:

```bash
pip install torch scikit-learn

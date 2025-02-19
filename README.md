# Neural Network Models and Data Loaders in PyTorch

This repository provides a collection of neural network models and custom data loaders implemented using PyTorch. It is designed for both beginners and advanced users to explore, experiment, and learn about neural network architectures and preprocessing pipelines.

---

## Table of Contents
1. [Models](#models)
    - [Linear Regression](#1-linear-regression)
    - [Multiclassification](#2-multiclassification)
    - [Handwritten Digits Classification Using Multilayer Perceptron](#3-handwritten-digits-classification-using-multilayer-perceptron)
    - [Convolutional Neural Network (CNN)](#4-convolutional-neural-network-cnn-coming-soon)
    - [Transfer Learning](#5-transfer-learning)
    - [Transformer](#6-transformer)
2. [Regularization](#regularization)
3. [Data Loaders](#data-loaders)
    - [Image Data Loader](#1-image-data-loader)
    - [Numerical Data Loader](#2-numerical-data-loader)
4. [Installation](#installation)
5. [Contributions](#contributions)
6. [License](#license)

---

## Models

This repository includes various neural network models, each organized in its own subdirectory for clarity.

### 1. **Linear Regression**
- **Description**: A basic implementation of linear regression for supervised learning tasks.
- **Directory**: [LinearRegression](linear_regression_pytorch)
- **Dependencies**: PyTorch
- **Features**:
  - Demonstrates the fundamental concepts of regression.
  - Uses mean squared error (MSE) as the loss function.
- **How to Run**:
  - Navigate to the `linear_regression_pytorch` directory.
  - Run the script:
    ```bash
    python linear_regression.py
    ```

---

### 2. **Multiclassification**
- **Description**: A neural network model for multiclass classification using the Iris dataset.
- **Directory**: [MultiClassification](multiclassification_pytorch)
- **Dependencies**: PyTorch, scikit-learn
- **Features**:
  - Two architectures: `Multiclassification` and `NeuralNetAdvance`.
  - Implements dropout for regularization.
  - Uses the Iris dataset for classification tasks.
- **How to Run**:
  - Navigate to the `multiclassification_pytorch` directory.
  - Run the script:
    ```bash
    python multiclassification.py
    ```

---

### 3. **Handwritten Digits Classification Using Multilayer Perceptron**
- **Description**: A neural network model for classifying handwritten digits using the MNIST dataset.
- **Directory**: [HandWritten_Digit_Classifier](handwritten_digits_pytorch)
- **Dependencies**: PyTorch, torchvision
- **Features**:
  - Implements a simple Multilayer Perceptron (MLP) architecture.
  - Trains on the MNIST dataset for handwritten digit classification.
  - Includes training and evaluation scripts.
- **How to Run**:
  - Navigate to the `handwritten_digits_pytorch` directory.
  - Run the script:
    ```bash
    python main.py
    ```
  - Example Output:
    ```
    Training Epoch: 1/10, Loss: 0.345, Accuracy: 92.3%
    Validation Accuracy: 91.8%
    ```

---

### 4. **Convolutional Neural Network (CNN)** (Coming Soon)
- **Description**: A model designed for processing grid-like data such as images using convolutional layers.
- **Directory**: [cnn_model_pytorch](cnn_model_pytorch)
- **Features**:
  - Convolutional, pooling, and fully connected layers.
  - Suitable for image classification and object detection tasks.

---

### 5. **Transfer Learning**
- **Description**: A model leveraging pre-trained deep learning models for efficient training on new datasets.
- **Directory**: [transfer_learning](transfer_learning_pytorch)
- **Dependencies**: PyTorch, torchvision
- **Features**:
  - Uses a pre-trained ResNet model to classify the Caltech101 dataset.
  - Demonstrates feature extraction and fine-tuning methods.
- **How to Run**:
  - Navigate to the `transfer_learning_pytorch` directory.
  - Run the script:
    ```bash
    python transfer_learning_caltech101.py
    ```

---

### 6. **Transformer**
- **Description**: Transformer models for NLP tasks using BERT.
- **Directory**: [transformer_pytorch](transformer_pytorch)
- **Dependencies**: PyTorch, transformers (Hugging Face library)
- **Features**:
  - Implements BERT-based spam detection.
  - Demonstrates basic BERT usage with fine-tuning capabilities.
- **How to Run**:
  - Navigate to the `transformer_pytorch` directory.
  - Run the script:
    ```bash
    python spam_detection_bert.py
    ```
  - To explore BERT basics, run:
    ```bash
    python bert_basics_usage.py
    ```

---

## Regularization

Regularization is a crucial technique in neural networks to prevent overfitting and improve the model’s generalization on unseen data. This repository incorporates several regularization methods in its models:

1. **Dropout**
   - Used in the `Multiclassification` model.
   - Randomly drops a fraction of neurons during training to prevent reliance on specific features.

2. **Weight Decay (L2 Regularization)**
   - Penalizes large weights by adding an L2 norm term to the loss function.
   - Can be configured in most models using PyTorch’s optimizer.

3. **Batch Normalization** 
   - Will be added in upcoming models like CNNs to stabilize learning and reduce dependence on initialization.

---

## Data Loaders

Custom data loaders to streamline data preprocessing and integration with PyTorch models.

### 1. **Image Data Loader**
- **Description**: A custom loader for loading and preprocessing image datasets from directory structures.
- **Directory**: [data_loader/image_data_loader.py](data_loader/image_data_loader.py)
- **Features**:
  - Loads images organized in class-specific subdirectories.
  - Applies transformations using `torchvision.transforms`.
  - Returns image-label pairs for PyTorch's `DataLoader`.

---

### 2. **Numerical Data Loader**
- **Description**: A custom loader for numerical datasets, such as tabular data.
- **Directory**: [data_loader/numerical_data_loader.py](data_loader/numerical_data_loader.py)

---

## Installation

### Prerequisites
Ensure you have the following installed:
- **Python 3.x**
- **PyTorch** (installation guide: [PyTorch](https://pytorch.org/get-started/locally/))
- **scikit-learn** (for preprocessing in multiclassification models)
- **torchvision** (for image transformations)

Install the necessary dependencies:
```bash
pip install torch torchvision scikit-learn transformers
```

---

## Contributions

Contributions are welcome! To contribute:
- Fork this repository.
- Create a feature branch (`git checkout -b feature-name`).
- Commit your changes (`git commit -m "Feature description"`).
- Push the branch (`git push origin feature-name`).
- Submit a pull request.

---

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


# Neural Network Models and Data Loaders in PyTorch

This repository includes various neural network models and custom data loaders implemented using PyTorch. The repository is modular and organized to help both beginners and advanced users explore and experiment with neural networks and custom data preprocessing pipelines.

---

## Models

This repository includes a range of neural network models, each implemented in its own subdirectory for clarity. Below are the currently available and planned models:

### 1. **Linear Regression**
- **Description**: A basic linear regression model demonstrating supervised learning.
- **Directory**: [LinearRegression](linear_regression_pytorch)
- **Dependencies**: PyTorch
- **How to Run**:
  - Navigate to the `linear_regression_pytorch` directory.
  - Run the script: `python linear_regression.py`.

---

### 2. **Multiclassification**
- **Description**: A neural network model for multiclass classification using the Iris dataset.
- **Directory**: [MultiClassification](multiclassification_pytorch)
- **Dependencies**: PyTorch, scikit-learn
- **How to Run**:
  - Navigate to the `multiclassification_pytorch` directory.
  - Run the script: `python multiclassification.py`.
- **Features**:
  - Two architectures: `Multiclassification` and `NeuralNetAdvance`.
  - Implements dropout for regularization.
  - Demonstrates the use of the Iris dataset for classification tasks.

---

### 3. **Convolutional Neural Network (CNN)** (Coming Soon)
- **Description**: A model designed to process grid-like data, such as images, by using convolutional layers.
- **Directory**: [cnn_model_pytorch](cnn_model_pytorch)

---

### 4. **Recurrent Neural Network (RNN)** (Coming Soon)
- **Description**: A type of neural network used for sequential data, such as time-series or natural language.
- **Directory**: [rnn_model_pytorch](rnn_model_pytorch)

---

### 5. **Autoencoder** (Coming Soon)
- **Description**: An unsupervised learning model used for dimensionality reduction and feature extraction.
- **Directory**: [autoencoder_model_pytorch](autoencoder_model_pytorch)

---

## Data Loaders

### 1. **Image Data Loader**
- **Description**: A custom data loader for loading and preprocessing image datasets from a directory structure.
- **Directory**: [data_loader/image_data_loader.py](data_loader/image_data_loader.py)
- **Features**:
  - Loads images from a directory with class-specific subdirectories.
  - Applies transformations using `torchvision.transforms`.
  - Returns image-label pairs for training/testing.
- **How to Use**:
  - Define the directory structure: `images/<class_name>/<image_file>`.
  - Instantiate the loader and pass it to a PyTorch `DataLoader`.
  - Example:
    ```python
    from data_loader.image_data_loader import ImageLoader
    from torch.utils.data import DataLoader
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageLoader(image_dir='images/train_data', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    for images, labels in train_loader:
        print(images.shape, labels)
        break
    ```

---

### 2. **Numerical Data Loader**
- **Description**: A custom data loader for numerical datasets (e.g., tabular data).
- **Directory**: [data_loader/numerical_data_loader.py](data_loader/numerical_data_loader.py)
- **Features**:
  - Accepts numerical data (e.g., features and labels).
  - Converts data into PyTorch tensors for compatibility with neural network models.
  - Suitable for datasets like CSV files or preprocessed numerical arrays.
- **How to Use**:
  - Pass the feature matrix and labels during initialization.
  - Example:
    ```python
    from data_loader.numerical_data_loader import CustomNumericalLoader
    from torch.utils.data import DataLoader

    X_train, Y_train = ...  # Load or preprocess your data

    train_dataset = CustomNumericalLoader(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for features, labels in train_loader:
        print(features.shape, labels)
        break
    ```

---

## Installation

### Prerequisites

Ensure you have the following installed on your machine:

- **Python** 3.x
- **PyTorch** (visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for installation instructions)
- **scikit-learn** (for preprocessing data in the Multiclassification model)
- **torchvision** (for image transformations in the Image Data Loader)

You can install the necessary dependencies using `pip`:

```bash
pip install torch torchvision scikit-learn
```

---

## Contributions

Contributions are welcome! If you'd like to add a new model or enhance the existing implementations, feel free to submit a pull request.

---

## License

This repository is licensed under the MIT License. See the LICENSE file for details.

---




import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os

# --- Configuration ---
DATASET_PATH = './caltech-101/'
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5
VALIDATION_SPLIT = 0.2 #Ratio of validation data

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Loading and Preprocessing Functions ---
def load_and_transform_data(dataset_path, image_size, validation_split):
    """
    Loads the dataset, applies transformations, and splits into training and validation sets.
    Args:
        dataset_path (str): Path to the dataset.
        image_size (tuple): Desired size of the images (height, width).
        validation_split (float): Ratio of data to use for validation (e.g., 0.2 for 20%).
    Returns:
        tuple: train_loader, test_loader, num_classes
    """
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Split into training and validation sets
    test_size = int(validation_split * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(full_dataset.classes)

    return train_loader, test_loader, num_classes

def display_sample_image(dataset_path,image_size):
    """
    Loads the dataset, applies transformations, and displays an image of it
    Args:
        dataset_path (str): Path to the dataset.
        image_size (tuple): Desired size of the images (height, width).
    Returns:
        void
    """
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Display a sample image and its label
    plt.figure(figsize=(2, 2))
    image, label = full_dataset[80]
    plt.imshow(image.permute(1, 2, 0))
    plt.title(full_dataset.classes[label])
    plt.show()

# --- Model Definitions ---
class CNN(nn.Module):
    """
    A simple CNN model for image classification.
    """
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3,3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Flatten(),
            nn.Linear(128*32*32, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, inputs):
        return self.network(inputs)

def create_resnet18_model(num_classes, pretrained=True):
    """
    Creates a ResNet18 model with transfer learning.
    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pre-trained weights.
    Returns:
        torchvision.models.resnet18: ResNet18 model.
    """
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify final layer
    return model

def create_efficientnet_b0_model(num_classes, pretrained=True):
    """
    Creates an EfficientNet-B0 model with transfer learning.
    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pre-trained weights.
    Returns:
        torchvision.models.efficientnet_b0: EfficientNet-B0 model.
    """
    model = models.efficientnet_b0(pretrained=pretrained)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Modify classifier layer
    return model

# --- Training and Validation Functions ---
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    """
    Trains the given model.
    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): DataLoader for the training set.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim): Optimizer.
        epochs (int): Number of epochs to train for.
        device (torch.device): Device to train on (CPU or GPU).
    """
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")


def validate_model(model, test_loader, device):
    """
    Validates the given model.
    Args:
        model (nn.Module): Model to validate.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to validate on (CPU or GPU).
    Returns:
        float: Accuracy on the test set.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy

# --- Main Execution ---
def main():
    """
    Main function to execute the training and validation of different models.
    """
    # 1. Load and Prepare Data
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at '{DATASET_PATH}'.  Please download the Caltech-101 dataset and place it in the correct directory.")
        return

    train_loader, test_loader, num_classes = load_and_transform_data(DATASET_PATH, IMAGE_SIZE, VALIDATION_SPLIT)
    print(f"Number of classes: {num_classes}")
    display_sample_image(DATASET_PATH,IMAGE_SIZE)

    # 2. Train CNN (without transfer learning)
    print("\n--- Training CNN without Transfer Learning ---")
    cnn_model = CNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    train_model(cnn_model, train_loader, criterion, optimizer, EPOCHS, device)
    validate_model(cnn_model, test_loader, device)

    # 3. Train ResNet18 (with transfer learning)
    print("\n--- Training ResNet18 with Transfer Learning ---")
    resnet_model = create_resnet18_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_model.parameters(), lr=LEARNING_RATE)
    train_model(resnet_model, train_loader, criterion, optimizer, EPOCHS, device)
    validate_model(resnet_model, test_loader, device)

    # 4. Train EfficientNet-B0 (with transfer learning)
    print("\n--- Training EfficientNet-B0 with Transfer Learning ---")
    efficientnet_model = create_efficientnet_b0_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(efficientnet_model.parameters(), lr=LEARNING_RATE)
    train_model(efficientnet_model, train_loader, criterion, optimizer, EPOCHS, device)
    validate_model(efficientnet_model, test_loader, device)

if __name__ == "__main__":
    main()

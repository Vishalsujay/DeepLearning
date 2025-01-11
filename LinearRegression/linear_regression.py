import torch
import torch.nn as nn

# Define the input and output data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Define the Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Define the loss function
def loss_function():
    return nn.MSELoss()

# Define the optimizer
def get_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == "__main__":
    # Initialize the model, loss function, and optimizer
    model = LinearRegression()
    criterion = loss_function()
    optimizer = get_optimizer(model)

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward pass
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log the loss
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")

    # Print model parameters
    print(f"Model Weights: {model.linear.weight.data}")
    print(f"Model Bias: {model.linear.bias.data}")

    # Test the model
    with torch.no_grad():
        tolerance = 0.1
        predicted = model(X)
        correct_predictions = torch.sum(torch.abs(predicted - Y) <= tolerance).item()
        total_predictions = Y.numel()
        accuracy = (correct_predictions / total_predictions) * 100

        # Display results
        print("=" * 20)
        print(f"Input Values: {X.squeeze().numpy()}")
        print(f"Predicted Values: {predicted.squeeze().numpy()}")
        print(f"Actual Values: {Y.squeeze().numpy()}")
        print(f"Accuracy: {accuracy:.2f}%")

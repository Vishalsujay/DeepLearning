import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.Category = df.Category.replace({'ham': 0, 'spam': 1})
    df_ham = df[df.Category == 0].sample(1000, random_state=42)
    df_spam = df[df.Category == 1]
    df_dataset = pd.concat([df_ham, df_spam])
    X_train, X_test, Y_train, Y_test = train_test_split(
        df_dataset.Message, df_dataset.Category, test_size=0.2, random_state=42
    )
    return X_train, X_test, Y_train, Y_test

# Tokenize data
def tokenize_data(tokenizer, texts, labels):
    embeddings = tokenizer(texts, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
    return embeddings['input_ids'], embeddings['attention_mask'], torch.tensor(labels, dtype=torch.float)

# Define model
class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask)
        sentence_embedding = bert_output.last_hidden_state[:, 0, :]
        return self.classifier(sentence_embedding)

# Training function
def train_model(model, train_dataloader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for input_ids, attention_mask, labels in train_dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            loss = criterion(output.squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}')

# Validation function
def validate_model(model, test_dataloader, device):
    model.eval()
    correct_predictions = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            predicted = (outputs >= 0.5).float().squeeze()
            correct_predictions += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct_predictions / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Main function
def main():
    # Load data
    file_path = 'spam.csv'  # Assuming spam.csv is in the same directory as the script
    X_train, X_test, Y_train, Y_test = load_and_preprocess_data(file_path)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize data
    train_input_ids, train_attention_mask, train_labels = tokenize_data(tokenizer, X_train.tolist(), Y_train.tolist())
    test_input_ids, test_attention_mask, test_labels = tokenize_data(tokenizer, X_test.tolist(), Y_test.tolist())

    # Create datasets
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = SentimentClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    # Training and Validation
    epochs = 5
    train_model(model, train_dataloader, optimizer, criterion, epochs, device)
    validate_model(model, test_dataloader, device)

if __name__ == "__main__":
    main()

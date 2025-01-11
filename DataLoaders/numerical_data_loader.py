import os
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_csv(datalist):
    
    data = pd.DataFrame(
        datalist,
        columns=['feature1', 'feature2', 'feature3', 'feature4', 'label']
    )
    data.to_csv('custom_data.csv', index=False)
    
    
#Create a Custom DataLoader for Numerical Data. This should implement 3 Key methods:
#- __init__ : Initialize the DataLoader
#- __len__ : Return the length of the dataset
#- __getitem__ : Return a sample from the dataset given an index
class CustomNumericalLoader(Dataset):
    #Initialize the DataLoader
    def __init__(self, data, label):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label.to_numpy(), dtype=torch.long)
        
    #Return the length of the dataset
    def __len__(self):
        return len(self.data)
    
    #Return a sample from the dataset given an index
    def __getitem__(self,idx):
        return self.data[idx], self.label[idx]
    


#Main Function
if __name__ == '__main__':
    datalist = [
        [0.374540, 0.950714, 0.731994, 0.598658, 0],
        [0.156019, 0.155995, 0.058084, 0.866176, 1],
        [0.601115, 0.708073, 0.020584, 0.969910, 0],
        [0.832443, 0.212339, 0.181825, 0.183405, 2],
        [0.304242, 0.524756, 0.431945, 0.291229, 1],
        [0.611853, 0.139494, 0.292145, 0.366362, 2],
        [0.456070, 0.785176, 0.199674, 0.514234, 1],
        [0.592415, 0.046450, 0.607545, 0.170524, 0],
        [0.065052, 0.948886, 0.965632, 0.808397, 2],
        [0.304614, 0.097672, 0.684233, 0.440152, 1],
    ]
    #Function call the create a CSV file
    create_csv(datalist)
    
    #load the csv file created 
    data = pd.read_csv('custom_data.csv')
    X = data.drop('label', axis=1)
    Y = data['label']
    
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    #Create data set objects for training
    train_dataset = CustomNumericalLoader(X_train, Y_train)
    test_dataset = CustomNumericalLoader(X_test, Y_test)
    
    # Create data loaders for training and testing
    train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)
    
    #Iterating over the batches of training and testing data
    for data_batch, label_batch in train_loader:
        print(data_batch, label_batch)
    
    
    
    
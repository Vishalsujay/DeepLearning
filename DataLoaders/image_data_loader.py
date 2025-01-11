import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image


#Create a Custom DataLoader for Image Data. This should implement 3 Key methods:
#- __init__ : Initialize the DataLoader
#- __len__ : Return the length of the dataset
#- __getitem__ : Return a sample from the dataset given an index
class ImageLoader(Dataset):
    #Initialize the DataLoader
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        #Load all the images from the path specified in the diectory
        for label, class_dir in enumerate(os.listdir(self.image_dir)):
            class_path = os.path.join(self.image_dir, class_dir)
            for file in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, file))
                self.labels.append(label)
    
    #Return the length of the dataset
    def __len__(self):
        return len(self.image_paths)
    
    #Return a sample from the dataset given an index
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        #Pytorch accepts images in the format Pillow format
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]
    
    



#Main Function
if __name__ == '__main__':
    #Give the patb to the train, test image paths
    train_path = "images/train_data/"
    test_path = "images/test_data/"

   
    
    #Define the transformations for the images
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    #Create Datasets
    train_image_dataset = ImageLoader(image_dir=train_path, transform=transform)
    test_image_dataset = ImageLoader(image_dir=test_path, transform=transform)
    
    #Create DataLoaders
    train_image_loader = DataLoader(dataset=train_image_dataset, batch_size=2, shuffle=True)
    test_image_loader = DataLoader(dataset=test_image_dataset, batch_size=2, shuffle=False)
    
    #Iterate over the data
    for images, labels in train_image_loader:
        print(images.shape, labels)
        break
    
    

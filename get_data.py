import torch
from torch import nn
import torch.nn.functional as F

# Dataloaders
from torch.utils.data import DataLoader

# Create dataset from images with alpha channel located in 'data' folder
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image


### Dataset example

class modCharDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(modCharDataset, self).__init__(root, transform, target_transform)
        self.samples = self.samples[:10]

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
def get_dataloaders(batch_size):
    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])
    
    train_dataset = modCharDataset(root='data', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader

# Test dataloader
train_loader = get_dataloaders(2)
for x in train_loader:
    print(x.shape)
    break
    

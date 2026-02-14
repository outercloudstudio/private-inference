"""
Binary Neural Network for MNIST
MLP with binarized weights based on https://arxiv.org/abs/1602.02830
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import numpy as np

from layers import BinaryLinear, Activation, BinaryActivation


def get_loaders(batch_size=128, data_dir='./data'):
    """
    Create train and validation data loaders for MNIST.
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to download/load MNIST data
    
    Returns:
        train_loader, val_loader
    """
    # Transform to normalize MNIST images
    # MNIST images are 28x28 grayscale, we'll flatten them to 784-dim vectors
    # and normalize to [-1, 1] range to match your binary network expectations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 -> 784
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data (used as validation)
    val_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader


class BinaryMLP(nn.Module):
    def __init__(self):
        super(BinaryMLP, self).__init__()
        
        self.flatten = nn.Flatten()
        
        # Binary layers
        self.fc1 = nn.Linear(784, 256)
        self.act1 = nn.ReLU()
        
        self.fc2 = nn.Linear(256, 256)
        self.act2 = nn.ReLU()
        
        self.fc3 = nn.Linear(256, 16)
        self.act3 = nn.ReLU()
        
        # Output layer (standard layer for final classification)
        self.fc4 = nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.act1(x)
        
        x = self.fc2(x)
        x = self.act2(x)
        
        x = self.fc3(x)
        x = self.act3(x)
        
        x = self.fc4(x)
        
        return x


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
        })
    
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            # print(data)
            # print(target)
            
            output = model(data)

            # print(output)
            # print("")

            loss = criterion(output, target)
            
            running_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
            })
    
    avg_loss = running_loss / len(val_loader)
    
    return avg_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_loaders(batch_size=256)
    
    model = BinaryMLP().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    
    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device, epoch)

        print(f"Epoch {epoch}/{epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f}")
    
    print("Training complete!")

    print(model.fc1.weight)
    print(model.fc1.bias)


if __name__ == "__main__":
    main()

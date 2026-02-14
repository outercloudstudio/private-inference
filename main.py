"""
Binary Neural Network for MNIST
MLP with binarized weights based on https://arxiv.org/abs/1602.02830
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

from layers import BinaryLinear, Activation

class ParityDataset(Dataset):
    """
    Dataset that generates binary vectors of -1 and 1, with labels indicating
    whether the count of 1s is even (0) or odd (1).
    """
    def __init__(self, size=10000, vector_length=8):
        """
        Args:
            size: Number of samples in the dataset
            vector_length: Length of each binary vector
        """
        self.size = size
        self.vector_length = vector_length
        
        self.data = torch.randint(0, 2, (size, vector_length)).float() * 2 - 1  # Convert to -1, 1
        
        count_ones = (self.data == 1).sum(dim=1)
        self.labels = (count_ones % 2).long()
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def get_parity_loaders(batch_size=256, train_size=50000, val_size=10000, vector_length=8):
    """
    Create train, validation, and test data loaders for the parity task.
    
    Args:
        batch_size: Batch size for data loaders
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        vector_length: Length of binary vectors
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = ParityDataset(size=train_size, vector_length=vector_length)
    val_dataset = ParityDataset(size=val_size, vector_length=vector_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class BinaryMLP(nn.Module):
    """
    Binary MLP architecture for MNIST.
    Architecture: 784 -> 2048 -> 2048 -> 2048 -> 10
    Uses binarized weights in hidden layers.
    """
    def __init__(self):
        super(BinaryMLP, self).__init__()
        
        self.fc1 = BinaryLinear(8, 8)
        self.act1 = Activation('relu')
        
        self.fc2 = BinaryLinear(8, 2)
        self.act2 = Activation('relu')
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        
        x = self.fc2(x)
        x = self.act2(x)
        
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
            
            output = model(data)
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
    
    train_loader, val_loader = get_parity_loaders(batch_size=256)
    
    model = BinaryMLP().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nStarting training...")
    for epoch in range(1, 20 + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device, epoch)

        print(f"Epoch {epoch}/{20} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f}")
    
    print("Training complete!")


if __name__ == "__main__":
    main()

"""
Binary Neural Network for MNIST
MLP with binarized weights based on https://arxiv.org/abs/1602.02830
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from data import get_loaders

from layers import BinaryLinear, Activation


class BinaryMLP(nn.Module):
    """
    Binary MLP architecture for MNIST.
    Architecture: 784 -> 2048 -> 2048 -> 2048 -> 10
    Uses binarized weights in hidden layers.
    """
    def __init__(self):
        super(BinaryMLP, self).__init__()
        
        self.flatten = nn.Flatten()
        
        # self.fc1 = BinaryLinear(784, 512)
        # self.act1 = Activation('relu')
        
        # self.fc2 = BinaryLinear(512, 512)
        # self.act2 = Activation('relu')
        
        # self.fc3 = BinaryLinear(512, 512)
        # self.act3 = Activation('relu')
        
        # self.fc4 = nn.Linear(512, 10)

        self.fc1 = BinaryLinear(196, 128)
        self.act1 = Activation('relu')
        
        self.fc2 = BinaryLinear(128, 128)
        self.act2 = Activation('relu')
        
        self.fc3 = BinaryLinear(128, 128)
        self.act3 = Activation('relu')
        
        self.fc4 = nn.Linear(128, 10)
    
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
        
        # Calculate metrics
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}'
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
    parser = argparse.ArgumentParser(description="Binary neural network for MNIST")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--log_dir", type=str, default="./logs/mnist_binary", help="TensorBoard log directory")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/mnist_binary.pt", help="Checkpoint path")
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, val_loader = get_loaders(batch_size=args.batch_size)
    
    # Create model
    model = BinaryMLP().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device, epoch)    

        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f}")
    
    print("Training complete!")

    torch.save(model.state_dict(), 'binary_model_small.pth')
    # torch.save(model.state_dict(), 'binary_model.pth')


if __name__ == "__main__":
    main()

"""
Binary Neural Network for MNIST
MLP with binarized weights based on https://arxiv.org/abs/1602.02830
"""

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_loaders
from model import Model, train_epoch, validate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_loaders(batch_size=256)
    
    model = Model().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    
    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device, epoch)

        print(f"Epoch {epoch}/{epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f}")
    
    print("Training complete!")

    torch.save(model.state_dict(), 'model.pth')


if __name__ == "__main__":
    main()

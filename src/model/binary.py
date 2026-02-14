"""
Binary Neural Network for MNIST
MLP with binarized weights based on https://arxiv.org/abs/1602.02830
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

from layers import BinaryLinear, Activation
from utils import get_mnist, ModelCheckpoint, load_model, count_parameters, calculate_accuracy
from tensorboard_logging import Logger


class BinaryMLP(nn.Module):
    """
    Binary MLP architecture for MNIST.
    Architecture: 784 -> 2048 -> 2048 -> 2048 -> 10
    Uses binarized weights in hidden layers.
    """
    def __init__(self):
        super(BinaryMLP, self).__init__()
        
        self.flatten = nn.Flatten()
        
        # Binary layers
        self.fc1 = BinaryLinear(784, 512)
        self.act1 = Activation('relu')
        
        self.fc2 = BinaryLinear(512, 512)
        self.act2 = Activation('relu')
        
        self.fc3 = BinaryLinear(512, 512)
        self.act3 = Activation('relu')
        
        # Output layer (standard layer for final classification)
        self.fc4 = nn.Linear(512, 10)
    
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


def train_epoch(model, train_loader, criterion, optimizer, device, logger, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        acc = calculate_accuracy(output, target)
        running_loss += loss.item()
        running_acc += acc
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.4f}'
        })
        
        # Log to TensorBoard
        global_step = epoch * len(train_loader) + batch_idx
        logger.log_scalar('batch/train_loss', loss.item(), global_step)
        logger.log_scalar('batch/train_acc', acc, global_step)
    
    avg_loss = running_loss / len(train_loader)
    avg_acc = running_acc / len(train_loader)
    
    return avg_loss, avg_acc


def validate(model, val_loader, criterion, device, epoch, logger):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            acc = calculate_accuracy(output, target)
            running_loss += loss.item()
            running_acc += acc
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })
    
    avg_loss = running_loss / len(val_loader)
    avg_acc = running_acc / len(val_loader)
    
    return avg_loss, avg_acc


def test(model, test_loader, device):
    """Test the model"""
    model.eval()
    running_acc = 0.0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            acc = calculate_accuracy(output, target)
            running_acc += acc
            
            pbar.set_postfix({'acc': f'{acc:.4f}'})
    
    avg_acc = running_acc / len(test_loader)
    return avg_acc


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
    train_loader, val_loader, test_loader = get_mnist(batch_size=args.batch_size)
    
    # Create model
    model = BinaryMLP().to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Logger and checkpoint
    logger = Logger(log_dir=args.log_dir)
    checkpoint = ModelCheckpoint(args.checkpoint, monitor='val_acc', mode='max')
    
    # Log hyperparameters
    logger.log_text('hyperparameters', str(vars(args)), 0)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, logger, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, logger)
        
        # Log epoch metrics
        logger.log_scalars('epoch/loss', {'train': train_loss, 'val': val_loss}, epoch)
        logger.log_scalars('epoch/accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save checkpoint
        checkpoint(model, val_acc, epoch)
    
    # Load best model and test
    print("\nLoading best model for testing...")
    model = load_model(model, args.checkpoint, device)
    test_acc = test(model, test_loader, device)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    # Log final metrics
    logger.log_hyperparams(
        {'lr': args.lr, 'batch_size': args.batch_size},
        {'test_accuracy': test_acc}
    )
    
    logger.close()
    print("Training complete!")


if __name__ == "__main__":
    main()

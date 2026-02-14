from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
        transforms.Resize((14, 14)),
        transforms.Lambda(lambda x: (x > 0.2).float()),
        transforms.Lambda(lambda x: x.view(-1))
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

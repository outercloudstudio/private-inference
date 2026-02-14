import torch
import torch.nn as nn
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(784, 256)
        self.act1 = nn.ReLU()
        
        self.fc2 = nn.Linear(256, 256)
        self.act2 = nn.ReLU()
        
        self.fc3 = nn.Linear(256, 16)
        self.act3 = nn.ReLU()
        
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

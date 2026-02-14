import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

class BalancingMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.quant = quantization.QuantStub()

        self.fc1 = nn.Linear(4, 4)
        
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(4, 2)
        
        self.dequant = quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)

        x = self.relu1(self.fc1(x))
        
        x = self.fc2(x)
        
        x = self.dequant(x)
        
        return x

def create_balancing_dataset(num_samples=10000):
    positions = np.random.uniform(-2.4, 2.4, num_samples)
    velocities = np.random.uniform(-3, 3, num_samples)
    angles = np.random.uniform(-0.3, 0.3, num_samples)
    angular_vels = np.random.uniform(-2, 2, num_samples)
    
    X = np.stack([positions, velocities, angles, angular_vels], axis=1)
    y = (angles > 0).astype(int)
    
    return TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))

def train_balancing_model():
    dataset = create_balancing_dataset()
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = BalancingMLP()
    
    model.train()
    model.qconfig = quantization.get_default_qat_qconfig('x86')
    quantization.prepare_qat(model, inplace=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(20):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/20 - Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%')
    
    model.eval()
    quantized_model = quantization.convert(model, inplace=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = quantized_model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    print(f'\nTest Accuracy: {acc:.2f}%')
    
    return quantized_model

model = train_balancing_model()

torch.save(model.state_dict(), 'balancing_quantized.pth')

print("\nModel saved!")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import json

class PolynomialActivation(nn.Module):
    """Swish-like polynomial approximation activation function
    
    Approximates a smooth activation similar to Swish/SiLU
    Formula: x * sigmoid_approx(x) where sigmoid is approximated by polynomial
    For FHE compatibility: uses low-degree polynomials
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.pow(x, 3)
    
class ScaleLinear(nn.Module):
    """Linear layer with integer weights and biases"""
    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale
        
    def forward(self, x):
        return torch.mul(x, self.scale)

class QuantizedLinear(nn.Module):
    """Linear layer with integer weights and biases"""
    def __init__(self, in_features, out_features, scale=1.0):
        super().__init__()
        self.weight_float = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_float = nn.Parameter(torch.randn(out_features))
        self.scale = scale
        self.quantized = False
        
    def forward(self, x):
        if self.quantized:
            return torch.matmul(torch.round(x), self.weight_int.t().float()) + self.bias_int.float()
        else:
            return torch.matmul(x, self.weight_float.t()) + self.bias_float
    
    def quantize(self):
        """Convert float weights to integers"""
        self.weight_int = torch.round(self.weight_float * self.scale).to(torch.int32)
        self.bias_int = torch.round(self.bias_float * self.scale).to(torch.int32)
        self.quantized = True

        self.weight_float.requires_grad = False
        self.bias_float.requires_grad = False

class BalancingMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale = ScaleLinear(scale=10.0)
        self.fc1 = QuantizedLinear(4, 4, scale=10.0)
        self.activation = PolynomialActivation()
        self.fc2 = QuantizedLinear(4, 2, scale=10.0)
        
    def forward(self, x):
        x = self.scale(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def quantize(self):
        self.fc1.quantize()
        self.fc2.quantize()

def create_balancing_dataset(num_samples=10000):
    positions = np.random.uniform(-2.4, 2.4, num_samples)
    velocities = np.random.uniform(-3, 3, num_samples)
    angles = np.random.uniform(-0.3, 0.3, num_samples)
    angular_vels = np.random.uniform(-2, 2, num_samples)
    
    X = np.stack([positions, velocities, angles, angular_vels], axis=1)
    y = (angles > 0).astype(int)
    
    return TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))

def train_balancing_model(model):
    dataset = create_balancing_dataset()
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model.train()
    
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
    
    print("\nQuantizing model to integer weights...")
    model.quantize()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    acc = 100. * correct / total
    print(f'Test Accuracy (with integer weights): {acc:.2f}%')

    print(model(torch.FloatTensor([0, 0, 2, 0])))

    return model

model = BalancingMLP()
trained_model = train_balancing_model(model)

torch.save(trained_model.state_dict(), 'balancing.pth')

weights_fc1 = model.fc1.weight_int.cpu().numpy().tolist()
bias_fc1 = model.fc1.bias_int.cpu().numpy().tolist()
weights_fc2 = model.fc2.weight_int.cpu().numpy().tolist()
bias_fc2 = model.fc2.bias_int.cpu().numpy().tolist()

model_params = {
    "fc1": {
        "weights": weights_fc1,
        "bias": bias_fc1
    },
    "fc2": {
        "weights": weights_fc2,
        "bias": bias_fc2
    },
    "scale": 10.0
}

with open('balancing_weights.json', 'w') as f:
    json.dump(model_params, f, indent=2)
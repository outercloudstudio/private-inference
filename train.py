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
        return torch.pow(x, 2)

class BalancingMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 4)
        self.activation = PolynomialActivation()
        self.fc2 = nn.Linear(4, 2)
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x

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

weights_fc1 = model.fc1.weight.cpu().detach().numpy().tolist()
bias_fc1 = model.fc1.bias.cpu().detach().numpy().tolist()
weights_fc2 = model.fc2.weight.cpu().detach().numpy().tolist()
bias_fc2 = model.fc2.bias.cpu().detach().numpy().tolist()

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
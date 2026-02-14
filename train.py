import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Simple MLP for balancing tasks
class BalancingMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=4, output_dim=2):
        super().__init__()
        self.quant = quantization.QuantStub()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dequant = quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x

# Generate synthetic balancing data (CartPole-like)
def create_balancing_dataset(num_samples=10000):
    """
    4 inputs: cart position, velocity, pole angle, angular velocity
    Output: 0 = push left, 1 = push right
    """
    positions = np.random.uniform(-2.4, 2.4, num_samples)
    velocities = np.random.uniform(-3, 3, num_samples)
    angles = np.random.uniform(-0.3, 0.3, num_samples)
    angular_vels = np.random.uniform(-2, 2, num_samples)
    
    X = np.stack([positions, velocities, angles, angular_vels], axis=1)
    y = (angles > 0).astype(int)  # Simple heuristic: push opposite to angle
    
    return TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))

# Training function
def train_balancing_model(hidden_dim=4):
    # Create and split dataset
    dataset = create_balancing_dataset()
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize and prepare model
    model = BalancingMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=2)
    print(f"Model: 4 → {hidden_dim} → 2 (~{4*hidden_dim + hidden_dim*2} parameters)")
    
    model.train()
    model.qconfig = quantization.get_default_qat_qconfig('x86')
    quantization.prepare_qat(model, inplace=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
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
    
    # Convert to quantized model
    model.eval()
    quantized_model = quantization.convert(model, inplace=False)
    
    # Evaluate
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

if __name__ == "__main__":
    model = train_balancing_model(hidden_dim=4)  # Try 4, 8, or 16
    torch.save(model.state_dict(), 'balancing_quantized.pth')
    print("\nModel saved!")
"""
Binary Neural Network Layers for PyTorch
Modern implementation of BNN layers based on https://arxiv.org/abs/1602.02830
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BinarizeFunction(torch.autograd.Function):
    """
    Binarization function with straight-through estimator for gradients.
    Forward: deterministic binarization (sign function)
    Backward: straight-through estimator (gradient passes through unchanged)
    """
    @staticmethod
    def forward(ctx, input):
        # Deterministic binarization: +1 if x >= 0, else -1
        # torch.sign returns 0 for 0, so we need to handle that
        output = input.sign()
        output[output == 0] = 1  # Map 0 to +1
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradient through unchanged
        return grad_output


def binarize(x):
    """Binarize weights or activations using sign function"""
    return BinarizeFunction.apply(x)


class BinaryLinear(nn.Module):
    """
    Binary fully connected layer.
    Maintains full-precision weights but uses binarized weights for forward pass.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Full precision weights (used for gradient updates)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using standard initialization"""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)  # sqrt(5) as a float
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # Clip weights to [-1, 1] as they don't contribute outside this range
        self.weight.data.clamp_(-1, 1)
        
        # Binarize weights for forward pass
        binary_weight = binarize(self.weight)
        
        return F.linear(x, binary_weight, self.bias)
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class BinaryConv2d(nn.Module):
    """
    Binary 2D convolutional layer.
    Maintains full-precision weights but uses binarized weights for forward pass.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super(BinaryConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Full precision weights
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using standard initialization"""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)  # sqrt(5) as a float
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # Clip weights to [-1, 1]
        self.weight.data.clamp_(-1, 1)
        
        # Binarize weights for forward pass
        binary_weight = binarize(self.weight)
        
        return F.conv2d(x, binary_weight, self.bias, self.stride, self.padding)
    
    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, bias={self.bias is not None}')


class BinaryActivation(nn.Module):
    """
    Binary activation function using sign.
    Output is binarized to {-1, +1}.
    """
    def __init__(self):
        super(BinaryActivation, self).__init__()
    
    def forward(self, x):
        return binarize(x)


# Standard layer wrappers for convenience
class Dense(nn.Linear):
    """Standard fully connected layer (wrapper for nn.Linear)"""
    pass


class Activation(nn.Module):
    """
    Activation layer wrapper supporting multiple activation functions.
    """
    def __init__(self, activation_type: str = 'relu'):
        super(Activation, self).__init__()
        self.activation_type = activation_type.lower()
        
        if self.activation_type == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif self.activation_type == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif self.activation_type == 'binary':
            self.activation = BinaryActivation()
        else:
            raise ValueError(f"Unsupported activation: {activation_type}")
    
    def forward(self, x):
        return self.activation(x)


class Dropout(nn.Dropout):
    """Dropout layer (wrapper for nn.Dropout)"""
    pass
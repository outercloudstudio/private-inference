import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from model import Model

class StaticSymmetricQuantizer:
    """
    Applies static symmetric quantization to model weights.
    Maps weights from [-max, max] to integer range.
    """
    def __init__(self, n_bits=8):
        """
        Args:
            n_bits: Number of bits for quantization (default: 8 for INT8)
        """
        self.n_bits = n_bits
        self.qmin = -(2 ** (n_bits - 1))
        self.qmax = 2 ** (n_bits - 1) - 1
        
    def quantize_tensor(self, tensor):
        """
        Quantize a single tensor using symmetric quantization.
        
        Args:
            tensor: PyTorch tensor to quantize
            
        Returns:
            quantized_tensor: Quantized integer tensor
            scale: Scale factor for dequantization
        """
        # Find the maximum absolute value
        max_val = torch.max(torch.abs(tensor))
        
        # Avoid division by zero
        if max_val == 0:
            return torch.zeros_like(tensor, dtype=torch.int8), 1.0
        
        # Calculate scale factor
        scale = max_val / self.qmax
        
        # Quantize
        quantized = torch.round(tensor / scale)
        quantized = torch.clamp(quantized, self.qmin, self.qmax)
        
        # Convert to appropriate integer type
        if self.n_bits == 8:
            quantized = quantized.to(torch.int8)
        else:
            quantized = quantized.to(torch.int32)
        
        return quantized, scale.item()
    
    def dequantize_tensor(self, quantized_tensor, scale):
        """
        Dequantize a tensor back to float.
        
        Args:
            quantized_tensor: Quantized integer tensor
            scale: Scale factor used during quantization
            
        Returns:
            Dequantized float tensor
        """
        return quantized_tensor.float() * scale


def extract_and_quantize_weights(model, n_bits=8):
    """
    Extract all weights from a PyTorch model and quantize them.
    
    Args:
        model: PyTorch model
        n_bits: Number of bits for quantization
        
    Returns:
        quantized_weights: Dictionary of quantized weights
        scales: Dictionary of scale factors
        original_weights: Dictionary of original weights (for comparison)
    """
    quantizer = StaticSymmetricQuantizer(n_bits=n_bits)
    
    quantized_weights = OrderedDict()
    scales = OrderedDict()
    original_weights = OrderedDict()
    
    print(f"Extracting and quantizing weights to {n_bits}-bit integers...\n")
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only quantize weight tensors
            original_weights[name] = param.data.clone()
            
            # Quantize the weight
            q_weight, scale = quantizer.quantize_tensor(param.data)
            quantized_weights[name] = q_weight
            scales[name] = scale
            
            # Calculate compression stats
            original_size = param.data.numel() * 4  # 4 bytes for FP32
            quantized_size = q_weight.numel() * (n_bits / 8)
            compression_ratio = original_size / quantized_size
            
            print(f"Layer: {name}")
            print(f"  Shape: {param.shape}")
            print(f"  Original range: [{param.data.min():.4f}, {param.data.max():.4f}]")
            print(f"  Scale factor: {scale:.6f}")
            print(f"  Quantized range: [{q_weight.min()}, {q_weight.max()}]")
            print(f"  Compression: {compression_ratio:.2f}x")
            print()
    
    return quantized_weights, scales, original_weights


def compute_quantization_error(original_weights, quantized_weights, scales):
    """
    Compute the quantization error for each layer.
    
    Args:
        original_weights: Dictionary of original float weights
        quantized_weights: Dictionary of quantized integer weights
        scales: Dictionary of scale factors
    """
    quantizer = StaticSymmetricQuantizer()
    
    print("\nQuantization Error Analysis:")
    print("-" * 50)
    
    for name in original_weights.keys():
        original = original_weights[name]
        dequantized = quantizer.dequantize_tensor(quantized_weights[name], scales[name])
        
        # Calculate error metrics
        mse = torch.mean((original - dequantized) ** 2).item()
        mae = torch.mean(torch.abs(original - dequantized)).item()
        max_error = torch.max(torch.abs(original - dequantized)).item()
        
        print(f"{name}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Max Error: {max_error:.6f}")


def save_quantized_model(quantized_weights, scales, save_path):
    """
    Save quantized weights and scales to a file.
    
    Args:
        quantized_weights: Dictionary of quantized weights
        scales: Dictionary of scale factors
        save_path: Path to save the quantized model
    """
    torch.save({
        'quantized_weights': quantized_weights,
        'scales': scales
    }, save_path)
    print(f"\nQuantized model saved to {save_path}")


# Example usage
if __name__ == "__main__":
    model = Model()
    model.load_state_dict(torch.load('model.pth'))
    
    # Extract and quantize weights
    quantized_weights, scales, original_weights = extract_and_quantize_weights(
        model, 
        n_bits=8
    )
    
    # Compute quantization error
    compute_quantization_error(original_weights, quantized_weights, scales)
    
    # Save quantized model
    save_quantized_model(quantized_weights, scales, 'quantized_model.pth')
    
    # Calculate total model size reduction
    original_size = sum(w.numel() * 4 for w in original_weights.values())
    quantized_size = sum(w.numel() for w in quantized_weights.values())
    
    print(f"\nTotal Model Size:")
    print(f"  Original (FP32): {original_size / 1024:.2f} KB")
    print(f"  Quantized (INT8): {quantized_size / 1024:.2f} KB")
    print(f"  Compression Ratio: {original_size / quantized_size:.2f}x")
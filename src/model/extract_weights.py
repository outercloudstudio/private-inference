import torch
import json
import numpy as np
from layers import binarize


def quantize(tensor):
    return torch.round(torch.mul(tensor, 100))


if __name__ == "__main__":
    state_dict = torch.load("binary_model.pth")

    data = {}

    data["fc1"] = {
        "weight": binarize(state_dict["fc1.weight"]).tolist(),
        "bias": torch.round(state_dict["fc1.bias"]).tolist(),
    }

    data["fc2"] = {
        "weight": binarize(state_dict["fc2.weight"]).tolist(),
        "bias": torch.round(state_dict["fc2.bias"]).tolist(),
    }

    data["fc3"] = {
        "weight": binarize(state_dict["fc3.weight"]).tolist(),
        "bias": torch.round(state_dict["fc3.bias"]).tolist(),
    }

    data["fc4"] = {
        "weight": quantize(state_dict["fc4.weight"]).tolist(),
        "bias": quantize(state_dict["fc4.bias"]).tolist(),
    }

    with open('binary_model.json', 'w') as f:
        json.dump(data, f)
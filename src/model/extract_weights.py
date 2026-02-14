import torch
import json
import numpy as np
from layers import binarize


def quantize(tensor):
    return torch.round(torch.mul(tensor, 100))


if __name__ == "__main__":
    state_dict = torch.load("binary_model_small.pth")

    data = {}

    data["fc1"] = {
        "weight": binarize(state_dict["fc1.weight"]).cpu().numpy().astype(np.int32).tolist(),
    }

    data["fc2"] = {
        "weight": binarize(state_dict["fc2.weight"]).cpu().numpy().astype(np.int32).tolist(),
    }

    data["fc3"] = {
        "weight": binarize(state_dict["fc3.weight"]).cpu().numpy().astype(np.int32).tolist(),
    }

    data["fc4"] = {
        "weight": quantize(state_dict["fc4.weight"]).cpu().numpy().astype(np.int32).tolist(),
        "bias": quantize(state_dict["fc4.bias"]).cpu().numpy().astype(np.int32).tolist(),
    }

    with open('binary_model.json', 'w') as f:
        json.dump(data, f)
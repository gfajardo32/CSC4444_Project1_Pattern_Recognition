"""Turn the NumPy MNIST arrays into PyTorch DataLoaders (shared by linear, MLP and CNN)."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X, y, batch_size=64, shuffle=False, flatten=True):
    """Wrap NumPy images and labels in a DataLoader that serves mini-batches.

    X: (N, 28, 28) float32 in [0, 1]   y: (N,) int64
    flatten=True  -> each batch of images is (B, 784)     (linear, MLP)
    flatten=False -> each batch of images is (B, 1, 28, 28) (CNN)
    returns: DataLoader yielding (images, labels)
    """
    # 1. Scale pixels from [0, 1] to [-1, 1]
    X_scaled = X * 2.0 - 1.0

    # 2. Reshape to (N, 784) or (N, 1, 28, 28) depending on flatten
    if flatten:
        X_reshaped = X_scaled.reshape(-1, 784)
    else:
        X_reshaped = X_scaled.reshape(-1, 1, 28, 28)

    # 3. Convert X and y to tensors with the exact expected dtypes
    X_tensor = torch.tensor(X_reshaped, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)

    # 4. Pair them with TensorDataset, then wrap in a DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

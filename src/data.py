from pathlib import Path
from PIL import Image
import numpy as np

def load_mnist(root= "MNIST", cache= "mnist_cache.npz"):
    """ This will retrun X (N, 28, 28) float 32 in [0,1]
    and y (N,) int labels. With no pytorch just numpy"""

    # Check if the cache file exists
    cache_path = None
    if cache is not None:
        cache_path = Path(root) / cache
        if cache_path.exists():
            print(f"Loading MNIST from cache: {cache_path}")
            data = np.load(cache_path)
            return data["X"], data["y"]

    # If the cache file does not exist, load the data from the original MNIST dataset
    valid_folders = {str(i) for i in range(10)}
    images, labels = [], []
    for Num_folder in sorted(Path(root).iterdir()):
        if Num_folder.name not in valid_folders:
            continue
        for file in sorted(Path(Num_folder).iterdir()):
            if file.suffix == ".png":
                images.append(np.array(Image.open(file)))
                labels.append(int(Num_folder.name))
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int64)

    # Save the cache so the next run can skip loading the PNGs
    if cache_path is not None:
        np.savez(cache_path, X=X, y=y)
        print(f"Saved MNIST to cache: {cache_path}")
    return X, y

def split(X, y, val_frac = 0.2, seed = 42):
    # shuffle the data with a fixed seed for reproducibility and return X_train and Y_train, X_val and Y_val
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split_idx = int(len(X) * (1 - val_frac))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]

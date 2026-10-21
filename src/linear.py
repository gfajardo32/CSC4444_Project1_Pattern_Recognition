"""Linear classifier y = Wx for MNIST, trained with L2 (MSE) loss and SGD (PyTorch)."""
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.data import load_mnist, split
from src.device import get_device
from src.knn import accuracy, confusion_matrix
from src.torch_data import make_loader


def build_model(bias=False):
    """Linear layer mapping 784 pixels to 10 digit scores. W has shape (10, 784)."""
    return nn.Linear(784, 10, bias=bias)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """One pass over the training data. returns: average loss over the epoch (float)."""
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Build one-hot float targets (B, 10) from integer labels
        targets = F.one_hot(labels.long(), num_classes=10).float()

        # The 5 PyTorch training steps:
        optimizer.zero_grad()
        scores = model(images)
        loss = loss_fn(scores, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Predict every image in loader. returns: (M,) NumPy array of predicted digits."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            scores = model(images)
            preds = torch.argmax(scores, dim=1)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


if __name__ == "__main__":
    torch.manual_seed(42)  # same starting weights every run
    device = get_device()

    X, y = load_mnist()
    X_train, y_train, X_val, y_val = split(X, y)

    # Hyperparameters
    batch_size = 64
    lr = 0.01  # 0.05 diverges to nan: [-1, 1] inputs have a large norm (background = -1)
    epochs = 20
    use_bias = False

    # Data loaders (make_loader flattens and scales X_train and X_val automatically)
    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True, flatten=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False, flatten=True)

    # Model, loss (MSE), and optimizer (SGD)
    model = build_model(bias=use_bias).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print("Training Linear Classifier (L2 Loss)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_preds = evaluate(model, val_loader, device)
        val_acc = accuracy(val_preds, y_val)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    runtime = time.time() - start_time
    
    # Final evaluation (reuse the last epoch's predictions)
    final_preds = val_preds
    final_acc = val_acc
    cm = confusion_matrix(final_preds, y_val)

    print(f"\nFinal Validation Accuracy: {final_acc:.4f} (Runtime: {runtime:.2f}s)")
    print("Confusion Matrix:\n", cm)

    # Visualize W: each row of model.weight reshaped to 28x28 in a 2x5 grid
    print("Saving weight visualizations...")
    weights = model.weight.detach().cpu().numpy()  # Shape: (10, 784)
    limit = np.abs(weights).max()  # symmetric color range so white = 0
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i in range(10):
        img = weights[i].reshape(28, 28)
        axes[i].imshow(img, cmap="coolwarm", vmin=-limit, vmax=limit)
        axes[i].set_title(f"Digit {i}")
        axes[i].axis("off")

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/linear_weights.png")
    print("Saved to results/figures/linear_weights.png")

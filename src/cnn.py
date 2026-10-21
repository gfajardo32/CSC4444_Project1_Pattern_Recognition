"""Convolutional neural network for MNIST, trained with cross-entropy loss (PyTorch).

Conv(1->32, 3x3) -> ReLU -> MaxPool(2) -> Conv(32->64, 3x3) -> ReLU -> MaxPool(2) -> Flatten -> Linear -> 10
"""
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.data import load_mnist, split
from src.device import get_device
from src.knn import accuracy, confusion_matrix
from src.linear import evaluate      # generic: any model that outputs (B, 10)
from src.mlp import train_one_epoch  # generic: cross-entropy with integer labels
from src.torch_data import make_loader


def conv_output_size(n, kernel, padding=0, stride=1):
    """Side length after a conv or pool layer: (n - kernel + 2*padding) // stride + 1."""
    return (n - kernel + 2 * padding) // stride + 1


def build_cnn(channels=(32, 64), kernel=3):
    """Two conv blocks (Conv -> ReLU -> MaxPool 2x2), then Flatten and a Linear layer to 10 scores.

    Input: (B, 1, 28, 28)   returns: nn.Module whose output is (B, 10) raw scores (no softmax)
    """
    c1, c2 = channels

    # Calculate spatial dimensions after each conv and 2x2 max-pool (kernel=2, stride=2)
    side = 28
    side = conv_output_size(side, kernel=kernel, padding=0, stride=1)  # 28 -> 26 (for kernel=3)
    side = conv_output_size(side, kernel=2, padding=0, stride=2)       # 26 -> 13
    side = conv_output_size(side, kernel=kernel, padding=0, stride=1)  # 13 -> 11
    side = conv_output_size(side, kernel=2, padding=0, stride=2)       # 11 -> 5

    return nn.Sequential(
        # Block 1
        nn.Conv2d(1, c1, kernel_size=kernel),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Block 2
        nn.Conv2d(c1, c2, kernel_size=kernel),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Classifier head
        nn.Flatten(),
        nn.Linear(c2 * side * side, 10)
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    device = get_device()

    X, y = load_mnist()
    X_train, y_train, X_val, y_val = split(X, y)

    # Hyperparameters
    channels = (32, 64)
    kernel = 3
    lr = 0.05
    batch_size = 64
    epochs = 10

    # Loaders with flatten=False -> batches of (B, 1, 28, 28)
    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True, flatten=False)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False, flatten=False)

    # Model, loss, and optimizer
    model = build_cnn(channels=channels, kernel=kernel).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    val_accs = []

    print(f"Training CNN (channels={channels}, kernel={kernel})...")
    start_time = time.time()

    val_preds, val_acc = np.array([], dtype=np.int64), 0.0  # filled in by the epoch loop
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_preds = evaluate(model, val_loader, device)
        val_acc = accuracy(val_preds, y_val)

        losses.append(avg_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    runtime = time.time() - start_time

    # Final evaluation (reusing last epoch's predictions and accuracy)
    final_preds = val_preds
    final_acc = val_accs[-1]
    cm = confusion_matrix(final_preds, y_val)

    print(f"\nFinal Validation Accuracy: {final_acc:.4f} (Runtime: {runtime:.2f}s)")
    print("Confusion Matrix:\n", cm)

    os.makedirs("results/figures", exist_ok=True)

    # 1. Figure: Learning curve -> results/figures/cnn_learning_curve.png
    print("Saving learning curve...")
    epochs_range = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_range, losses, marker="o")
    ax1.set_title("CNN Training Loss vs. Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True)

    ax2.plot(epochs_range, val_accs, marker="o", color="green")
    ax2.set_title("CNN Validation Accuracy vs. Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("results/figures/cnn_learning_curve.png")
    plt.close(fig)
    print("Saved to results/figures/cnn_learning_curve.png")

    # 2. Figure: First-layer filters -> results/figures/cnn_filters.png
    print("Saving first-layer filters...")
    first_conv = model[0]
    assert isinstance(first_conv, nn.Conv2d)
    filters = first_conv.weight.detach().cpu().numpy()  # Shape: (c1, 1, kernel, kernel)
    num_filters = filters.shape[0]
    limit = np.abs(filters).max()  # one symmetric color range for all filters, so white = 0
    cols = 8
    rows = (num_filters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 1.5 * rows))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < num_filters:
            axes[i].imshow(filters[i, 0], cmap="coolwarm", vmin=-limit, vmax=limit)
            axes[i].set_title(f"F{i}", fontsize=9)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("results/figures/cnn_filters.png")
    plt.close(fig)
    print("Saved to results/figures/cnn_filters.png")

"""Multilayer perceptron for MNIST: 784 -> 256 -> 128 -> 10 with ReLU, trained with SGD (PyTorch)."""
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.data import load_mnist, split
from src.device import get_device
from src.knn import accuracy, confusion_matrix
from src.linear import evaluate  # already generic: works for any model that outputs (B, 10) scores
from src.torch_data import make_loader


def build_mlp(hidden_sizes=(256, 128)):
    """Stack of Linear -> ReLU layers, then a final Linear to 10 digit scores.

    hidden_sizes=(256, 128) gives 784 -> 256 -> 128 -> 10.
    returns: nn.Module whose output is (B, 10) raw scores (no softmax at the end)
    """
    layers = []
    in_size = 784

    for h_size in hidden_sizes:
        layers.append(nn.Linear(in_size, h_size))
        layers.append(nn.ReLU())
        in_size = h_size

    # Final output layer: raw scores (logits), no activation after it
    layers.append(nn.Linear(in_size, 10))

    return nn.Sequential(*layers)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """One pass over the training data. returns: average loss over the epoch (float).

    Same 5 steps as src/linear.py, but cross-entropy takes the integer labels directly,
    so there is no one-hot step.
    """
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        scores = model(images)
        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = get_device()

    X, y = load_mnist()
    X_train, y_train, X_val, y_val = split(X, y)

    # Hyperparameters
    hidden_sizes = (256, 128)
    lr = 0.05
    batch_size = 64
    epochs = 15

    # Data loaders (flatten=True, same as the linear classifier)
    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True, flatten=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False, flatten=True)

    # Model, loss (CrossEntropyLoss), and optimizer (SGD)
    model = build_mlp(hidden_sizes=hidden_sizes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    val_accs = []

    print(f"Training MLP {784} -> {' -> '.join(map(str, hidden_sizes))} -> 10...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_preds = evaluate(model, val_loader, device)
        val_acc = accuracy(val_preds, y_val)

        losses.append(avg_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    runtime = time.time() - start_time

    # Final accuracy and confusion matrix (reuse the last epoch's predictions)
    final_preds = val_preds
    final_acc = val_acc
    cm = confusion_matrix(final_preds, y_val)

    print(f"\nFinal Validation Accuracy: {final_acc:.4f} (Runtime: {runtime:.2f}s)")
    print("Confusion Matrix:\n", cm)

    # Plot learning curve (loss and validation accuracy vs epoch)
    print("Saving learning curve...")
    epochs_range = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_range, losses, marker="o")
    ax1.set_title("Training Loss vs. Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True)

    ax2.plot(epochs_range, val_accs, marker="o", color="green")
    ax2.set_title("Validation Accuracy vs. Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/mlp_learning_curve.png")
    print("Saved to results/figures/mlp_learning_curve.png")

import numpy as np
import matplotlib.pyplot as plt
from src.data import load_mnist, split
from src.knn import accuracy
import os


def binarize(X):
    """Convert pixel values to 0 or 1 based on a 0.5 threshold.
    
    X: (N, 784) float array in [0, 1]
    returns: (N, 784) integer array of 0s and 1s
    """
    return (X > 0.5).astype(int)

def fit(X_bin, y, alpha=1):
    """Train the Naive Bayes classifier using Laplace smoothing.
    
    X_bin: (N, 784) binarized training images (0s and 1s)
    y: (N,) training labels
    alpha: Laplace smoothing parameter
    
    returns:
        priors: (10,) array of class probabilities P(y=c)
        pixel_probs: (10, 784) array of pixel probabilities P(x_i=1 | y=c)
    """
    N, D = X_bin.shape  # number of images, number of pixels per image
    priors = np.zeros(10)
    pixel_probs = np.zeros((10, D))
    
    for c in range(10):
        # Select only the images for this digit
        X_c = X_bin[y == c]
        N_c = X_c.shape[0]
        
        # 1. Calculate prior: fraction of images that are this digit
        priors[c] = N_c / N
        
        # 2. Calculate pixel probabilities with Laplace smoothing
        # X_c.sum(axis=0) counts how many times each pixel is 1 for this digit
        pixel_counts = X_c.sum(axis=0)
        pixel_probs[c] = (pixel_counts + alpha) / (N_c + 2 * alpha)
        
    return priors, pixel_probs

def predict_loop(X_bin, priors, probs):
    """Predict digits using Naive Bayes log-probabilities (Loop).
    
    X_bin: (M, 784) binarized validation images
    priors: (10,) class probabilities
    probs: (10, 784) pixel probabilities
    
    returns: (M,) array of predicted labels
    """
    M = X_bin.shape[0]
    predictions = np.empty(M, dtype=int)
    
    # Pre-calculate logs to avoid repeating it M times
    log_priors = np.log(priors)
    log_probs = np.log(probs)
    log_inv_probs = np.log(1 - probs)
    
    for i in range(M):
        # Element-wise multiplication, then sum across all 784 pixels
        # X_bin[i] is (784,), log_probs is (10, 784). 
        # Broadcasting applies the single image across all 10 digit probability templates.
        on_score = np.sum(X_bin[i] * log_probs, axis=1)
        off_score = np.sum((1 - X_bin[i]) * log_inv_probs, axis=1)
        
        # Total score for this image across all 10 digits
        scores = log_priors + on_score + off_score
        predictions[i] = np.argmax(scores)
        
    return predictions

if __name__ == "__main__":
    X, y = load_mnist()
    X_train, y_train, X_val, y_val = split(X, y)

    X_train_flat = X_train.reshape(-1, 784)
    X_val_flat = X_val.reshape(-1, 784)

    X_train_bin = binarize(X_train_flat)
    X_val_bin = binarize(X_val_flat)

    print("Training Naive Bayes...")
    priors, probs = fit(X_train_bin, y_train, alpha=1)

    print("Predicting...")
    preds = predict_loop(X_val_bin, priors, probs)

    acc = accuracy(preds, y_val)
    print(f"Validation Accuracy: {acc:.4f}")

    print("Saving probability maps...")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i in range(10):
        img = probs[i].reshape(28, 28)
        axes[i].imshow(img, cmap='viridis')
        axes[i].set_title(f"Digit {i}")
        axes[i].axis('off')
        
    plt.tight_layout()
    
    # Ensure directory exists and save
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/naive_bayes_maps.png")
    print("Saved to results/figures/naive_bayes_maps.png")



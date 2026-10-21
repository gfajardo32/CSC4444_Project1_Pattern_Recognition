"""Pick the fastest available PyTorch device: NVIDIA GPU, Apple GPU, or CPU."""
import torch


def get_device():
    """Return a torch.device for CUDA if available, else Apple MPS, else CPU."""
    if torch.cuda.is_available():          # NVIDIA GPU (Windows/Linux)
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple Silicon GPU (macOS)
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


if __name__ == "__main__":
    get_device()

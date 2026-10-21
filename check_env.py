"""Sanity check for the project environment. Run: .venv/bin/python check_env.py"""
from pathlib import Path

import matplotlib
import numpy as np
import PIL
import torch
import torchvision
from PIL import Image

print(f"numpy       {np.__version__}")
print(f"pillow      {PIL.__version__}")
print(f"matplotlib  {matplotlib.__version__}")
print(f"torch       {torch.__version__}")
print(f"torchvision {torchvision.__version__}")
print(f"MPS (Apple GPU) available: {torch.backends.mps.is_available()}")

data = Path(__file__).parent / "MNIST"
counts = {d.name: len(list(d.glob("*.png"))) for d in sorted(data.iterdir()) if d.is_dir()}
print(f"images per digit: {counts}  total={sum(counts.values())}")
sample = next((data / "0").glob("*.png"))
arr = np.array(Image.open(sample))
print(f"sample {sample.name}: shape={arr.shape} dtype={arr.dtype} range=[{arr.min()}, {arr.max()}]")

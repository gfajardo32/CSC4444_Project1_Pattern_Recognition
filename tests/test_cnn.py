"""Tests for src/cnn.py. Tiny fake data on the CPU."""
import numpy as np
import torch
from torch import nn

from src.torch_data import make_loader
from src.cnn import conv_output_size, build_cnn
from src.mlp import train_one_epoch
from src.linear import evaluate
from tests.test_linear import fake_data, DEVICE


# ---------- conv_output_size ----------

def test_conv_output_size_known_cases():
    # 28 with kernel 3 (no padding, stride 1): (28 - 3 + 0) // 1 + 1 = 26
    assert conv_output_size(28, kernel=3, padding=0, stride=1) == 26
    
    # 28 with kernel 3, padding 1, stride 1: (28 - 3 + 2) // 1 + 1 = 28
    assert conv_output_size(28, kernel=3, padding=1, stride=1) == 28
    
    # 26 with MaxPool kernel 2, stride 2: (26 - 2 + 0) // 2 + 1 = 13
    assert conv_output_size(26, kernel=2, padding=0, stride=2) == 13
    
    # 11 with MaxPool kernel 2, stride 2: (11 - 2 + 0) // 2 + 1 = 9 // 2 + 1 = 5
    # (The leftover 11th row/column is dropped because a 2x2 window cannot fit)
    assert conv_output_size(11, kernel=2, padding=0, stride=2) == 5


# ---------- build_cnn ----------

def test_build_cnn_output_shape():
    model = build_cnn()
    X_batch = torch.randn(5, 1, 28, 28)
    out = model(X_batch)
    
    assert out.shape == (5, 10)


def test_build_cnn_parameter_count():
    # Hand calculation for channels=(32, 64), kernel=3:
    # Spatial size progression: 28 -> 26 (conv1) -> 13 (pool1) -> 11 (conv2) -> 5 (pool2)
    # 1. Conv2d(1, 32, 3):   (1 * 32 * 3 * 3) + 32  =    320
    # 2. Conv2d(32, 64, 3):  (32 * 64 * 3 * 3) + 64 = 18,496
    # 3. Linear(64*5*5, 10): (1600 * 10) + 10       = 16,010
    # Total: 320 + 18,496 + 16,010 = 34,826
    model = build_cnn(channels=(32, 64), kernel=3)
    total_params = sum(p.numel() for p in model.parameters())
    
    expected = (1 * 32 * 3 * 3 + 32) + (32 * 64 * 3 * 3 + 64) + (64 * 5 * 5 * 10 + 10)
    assert expected == 34826
    assert total_params == expected


def test_build_cnn_has_two_conv_layers_with_relu_and_pool():
    model = build_cnn()
    
    conv_count = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    relu_count = sum(1 for m in model.modules() if isinstance(m, nn.ReLU))
    pool_count = sum(1 for m in model.modules() if isinstance(m, nn.MaxPool2d))
    
    assert conv_count == 2
    assert relu_count == 2
    assert pool_count == 2
    assert isinstance(list(model.children())[-1], nn.Linear)


def test_build_cnn_custom_channels():
    model = build_cnn(channels=(8, 16), kernel=3)
    X_batch = torch.randn(4, 1, 28, 28)
    out = model(X_batch)
    
    assert out.shape == (4, 10)


# ---------- training and evaluation ----------

def test_train_one_epoch_lowers_loss_cnn():
    torch.manual_seed(0)
    X, y = fake_data(n=8)
    loader = make_loader(X, y, batch_size=4, flatten=False, shuffle=False)
    
    model = build_cnn(channels=(4, 8), kernel=3).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    
    first_loss = train_one_epoch(model, loader, loss_fn, optimizer, DEVICE)
    last_loss = first_loss
    for _ in range(5):
        last_loss = train_one_epoch(model, loader, loss_fn, optimizer, DEVICE)
        
    assert last_loss < first_loss


def test_evaluate_works_with_cnn():
    X, y = fake_data(n=8)
    loader = make_loader(X, y, batch_size=4, flatten=False, shuffle=False)
    model = build_cnn(channels=(4, 8), kernel=3).to(DEVICE)
    
    preds = evaluate(model, loader, DEVICE)
    
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (8,)
    assert np.all((preds >= 0) & (preds <= 9))

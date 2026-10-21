"""Tests for src/mlp.py. Tiny fake data on the CPU."""
import numpy as np
import torch
from torch import nn

from src.torch_data import make_loader
from src.mlp import build_mlp, train_one_epoch
from src.linear import evaluate
from tests.test_linear import fake_data, DEVICE


# ---------- build_mlp ----------

def test_build_mlp_output_shape():
    model = build_mlp()
    X_batch = torch.randn(5, 784)
    out = model(X_batch)
    
    assert out.shape == (5, 10)


def test_build_mlp_parameter_count():
    # Hand calculation for 784 -> 256 -> 128 -> 10:
    # Layer 1 (784 -> 256): 784 * 256 + 256 = 200,960
    # Layer 2 (256 -> 128): 256 * 128 + 128 =  32,896
    # Layer 3 (128 -> 10):  128 * 10  + 10  =   1,290
    # Total: 200,960 + 32,896 + 1,290 = 235,146
    model = build_mlp(hidden_sizes=(256, 128))
    total_params = sum(p.numel() for p in model.parameters())
    
    expected = (784 * 256 + 256) + (256 * 128 + 128) + (128 * 10 + 10)
    assert expected == 235146
    assert total_params == expected


def test_build_mlp_custom_hidden_sizes():
    # Hand calculation for 784 -> 32 -> 10:
    # Layer 1 (784 -> 32): 784 * 32 + 32 = 25,120
    # Layer 2 (32 -> 10):  32 * 10  + 10 =    330
    # Total: 25,450
    model = build_mlp(hidden_sizes=(32,))
    X_batch = torch.randn(4, 784)
    out = model(X_batch)
    
    total_params = sum(p.numel() for p in model.parameters())
    expected = (784 * 32 + 32) + (32 * 10 + 10)
    
    assert out.shape == (4, 10)
    assert total_params == expected


def test_build_mlp_has_relu_between_layers():
    hidden_sizes = (256, 128)
    model = build_mlp(hidden_sizes=hidden_sizes)
    
    relu_count = sum(1 for m in model.modules() if isinstance(m, nn.ReLU))
    assert relu_count == len(hidden_sizes)
    assert isinstance(list(model.children())[-1], nn.Linear)


# ---------- train_one_epoch ----------

def test_train_one_epoch_lowers_loss():
    torch.manual_seed(0)
    X, y = fake_data(n=8)
    loader = make_loader(X, y, batch_size=4, flatten=True, shuffle=False)
    
    model = build_mlp(hidden_sizes=(16,)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    first_loss = train_one_epoch(model, loader, loss_fn, optimizer, DEVICE)
    for _ in range(5):
        last_loss = train_one_epoch(model, loader, loss_fn, optimizer, DEVICE)
        
    assert last_loss < first_loss


def test_train_one_epoch_uses_integer_labels():
    torch.manual_seed(0)
    X, y = fake_data(n=8)
    loader = make_loader(X, y, batch_size=4, flatten=True, shuffle=False)
    
    model = build_mlp(hidden_sizes=(16,)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    
    loss = train_one_epoch(model, loader, loss_fn, optimizer, DEVICE)
    
    assert isinstance(loss, float)
    assert np.isfinite(loss)


# ---------- evaluate (reused from linear) ----------

def test_evaluate_works_with_mlp():
    X, y = fake_data(n=8)
    loader = make_loader(X, y, batch_size=4, flatten=True, shuffle=False)
    model = build_mlp(hidden_sizes=(16,)).to(DEVICE)
    
    preds = evaluate(model, loader, DEVICE)
    
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (8,)
    assert np.all((preds >= 0) & (preds <= 9))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def compute_l1_penalty(model: nn.Module, device: str, lambda_: float = 1e-4) -> torch.Tensor:
    penalty = torch.tensor(0.0, device = device)
    for p in model.parameters():
        penalty += p.abs().sum()       # |w| summieren
    return lambda_ * penalty


def compute_l2_penalty(model: nn.Module, device: str ,lambda_: float = 1e-4) -> torch.Tensor:
    penalty = torch.tensor(0.0, device = device)
    for p in model.parameters():
        penalty += p.pow(2).sum()      # w² summieren
    return lambda_ * penalty


REGULARIZER_FN = {
    None:    lambda m: 0.0,                                    # keine Regularisierung
    "l1":    compute_l1_penalty,                               # nur L1
    "l2":    compute_l2_penalty,                               # nur L2
    "l1_l2": lambda m: compute_l1_penalty(m) + compute_l2_penalty(m),  # beide
}

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  optim.Optimizer,
    regularizer: str | None,
    DEVICE: str
) -> tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    reg_fn = REGULARIZER_FN[regularizer]

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb) + reg_fn(model, DEVICE)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(xb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(xb)

    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss   = criterion(logits, yb)

        total_loss += loss.item() * len(xb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(xb)

    return total_loss / n, correct / n

def check_device():
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print(f"Verwende Gerät: {DEVICE}")
    return DEVICE
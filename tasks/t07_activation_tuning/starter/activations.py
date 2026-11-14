"""Starter activations: plain ReLU on every hidden layer (0 activation parameters)."""
import torch.nn as nn


def build_activations(width: int) -> list[nn.Module]:
    return [nn.ReLU() for _ in range(3)]

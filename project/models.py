"""Definiert die MLP-Modellarchitektur für die Bildklassifikation."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Mehrschichtiges Perzeptron mit variabler Tiefe, Aktivierung und Dropout."""

    def __init__(self, input_size, n_layers, layer_sizes, activation, dropout_rate):
        """Erzeugt die Layerstruktur des MLP basierend auf den Hyperparametern."""
        super().__init__()
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
        }

        layers = [nn.Flatten()]
        in_features = input_size

        for i in range(n_layers):
            layers.append(nn.Linear(in_features, layer_sizes[i]))
            layers.append(activation_map[activation])
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = layer_sizes[i]

        layers.append(nn.Linear(in_features, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Berechnet den Vorwärtsdurchlauf für einen Eingabebatch."""
        return self.net(x)

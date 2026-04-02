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


class SimpleCNN(nn.Module):
    """Einfaches CNN mit Optuna-kompatibler MLP-Head-Signatur."""

    def __init__(
        self,
        input_size=None,
        n_layers: int = 1,
        layer_sizes: list[int] | None = None,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        num_classes: int = 10,
    ):
        super().__init__()

        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
        }
        if activation not in activation_map:
            raise ValueError(f'Unbekannte Aktivierung: {activation}')

        if layer_sizes is None:
            layer_sizes = [128] * n_layers
        if n_layers != len(layer_sizes):
            raise ValueError('n_layers muss der Länge von layer_sizes entsprechen.')

        _ = input_size

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        head_layers = []
        in_features = 64
        for hidden_size in layer_sizes:
            head_layers.append(nn.Linear(in_features, hidden_size))
            head_layers.append(activation_map[activation])
            if dropout_rate > 0.0:
                head_layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size

        head_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
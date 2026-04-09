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
    """Einfaches CNN mit Optuna-kompatiblen CNN- und MLP-Head-Hyperparametern."""

    def __init__(
        self,
        input_size=None,
        n_layers: int = 1,
        layer_sizes: list[int] | None = None,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        num_classes: int = 10,
        conv_channels: tuple[int, int, int] = (32, 64, 128),
        kernel_size: int = 3,
        pool_type: str = 'max',
        use_batchnorm: bool = False,
        cnn_dropout_rate: float = 0.0,
    ):
        super().__init__()

        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
        }
        if activation not in activation_map:
            raise ValueError(f'Unbekannte Aktivierung: {activation}')

        if layer_sizes is None:
            layer_sizes = [128] * n_layers
        if n_layers != len(layer_sizes):
            raise ValueError('n_layers muss der Länge von layer_sizes entsprechen.')

        if len(conv_channels) != 3:
            raise ValueError('conv_channels muss genau drei Werte enthalten.')
        if kernel_size not in (3, 5):
            raise ValueError('kernel_size muss 3 oder 5 sein.')
        if pool_type not in ('max', 'avg'):
            raise ValueError("pool_type muss 'max' oder 'avg' sein.")

        _ = input_size

        activation_cls = activation_map[activation]
        padding = kernel_size // 2

        def make_activation(inplace: bool = False) -> nn.Module:
            if activation == 'relu':
                return activation_cls(inplace=inplace)
            return activation_cls()

        def pool_layer() -> nn.Module:
            if pool_type == 'max':
                return nn.MaxPool2d(kernel_size=2)
            return nn.AvgPool2d(kernel_size=2)

        feature_layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in conv_channels:
            feature_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            if use_batchnorm:
                feature_layers.append(nn.BatchNorm2d(out_channels))
            feature_layers.append(make_activation(inplace=True))
            feature_layers.append(pool_layer())
            if cnn_dropout_rate > 0.0:
                feature_layers.append(nn.Dropout2d(cnn_dropout_rate))
            in_channels = out_channels

        feature_layers.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ])
        self.features = nn.Sequential(*feature_layers)

        head_layers: list[nn.Module] = []
        in_features = conv_channels[-1]
        for hidden_size in layer_sizes:
            head_layers.append(nn.Linear(in_features, hidden_size))
            head_layers.append(make_activation())
            if dropout_rate > 0.0:
                head_layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size

        head_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
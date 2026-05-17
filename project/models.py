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
    """CNN mit fix 5 Conv-Layern und variablem FC-Head (gleiche FC-Groesse)."""

    def __init__(
        self,
        input_size=None,
        n_fc_layers: int = 3,             # wird per Optuna optimiert
        fc_hidden_size: int = 128,        # wird per Optuna optimiert
        activation: str = "relu",         # relu | elu | swish
        dropout_rate: float = 0.0,
        num_classes: int = 10,
        conv_channels: tuple[int, int, int, int, int] = (32, 64, 128, 256, 512),  # fix
        kernel_size: int = 3,
        pool_type: str = "max",
        use_batchnorm: bool = False,
        cnn_dropout_rate: float = 0.0,
    ):
        super().__init__()

        if activation not in ("relu", "elu", "swish"):
            raise ValueError(f"Unbekannte Aktivierung: {activation}")
        if len(conv_channels) != 5:
            raise ValueError("conv_channels muss genau 5 Werte enthalten.")
        if n_fc_layers < 1:
            raise ValueError("n_fc_layers muss >= 1 sein.")

        _ = input_size
        padding = kernel_size // 2

        def make_activation(inplace: bool = False) -> nn.Module:
            if activation == "relu":
                return nn.ReLU(inplace=inplace)
            if activation == "elu":
                return nn.ELU(inplace=inplace)
            return nn.SiLU(inplace=inplace)  # Swish

        def pool_layer() -> nn.Module:
            return nn.MaxPool2d(2) if pool_type == "max" else nn.AvgPool2d(2)

        feature_layers: list[nn.Module] = []
        in_channels = 3
        for i, out_channels in enumerate(conv_channels):
            feature_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            if use_batchnorm:
                feature_layers.append(nn.BatchNorm2d(out_channels))
            feature_layers.append(make_activation(inplace=True))
            if i < len(conv_channels) - 1:  # 5 Conv, Pool nur in den ersten 4
                feature_layers.append(pool_layer())
            if cnn_dropout_rate > 0.0:
                feature_layers.append(nn.Dropout2d(cnn_dropout_rate))
            in_channels = out_channels

        feature_layers.extend([nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()])
        self.features = nn.Sequential(*feature_layers)

        head_layers: list[nn.Module] = []
        in_features = conv_channels[-1]
        for _ in range(n_fc_layers):
            head_layers.append(nn.Linear(in_features, fc_hidden_size))
            head_layers.append(make_activation())
            if dropout_rate > 0.0:
                head_layers.append(nn.Dropout(dropout_rate))
            in_features = fc_hidden_size

        head_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))



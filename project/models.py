import torch.nn as nn
import torch
"""
class MLP(nn.Module):
    def __init__(self, layer_sizes, activation="relu", dropout_rate=0.2):
        super().__init__()

        act = nn.ReLU if activation == "relu" else nn.GELU
        layers = [nn.Flatten()]

        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]
            layers.append(nn.Linear(in_features, out_features))

            is_last = i == len(layer_sizes) - 2
            if not is_last:
                layers.append(act())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
"""

class MLP(nn.Module):
    def __init__(self, input_size, n_layers, layer_sizes, activation, dropout_rate):
        super().__init__()
        activation_map = {
            "relu":    nn.ReLU(),
            "tanh":    nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
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
        return self.net(x)
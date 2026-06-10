from __future__ import annotations

import torch
import torch.nn as nn


class CNNDQN(nn.Module):
    """
    CNN-based Deep Q-Network for Snake.

    Input:
        Grid state from SnakeEnv.get_state_grid()
        Shape: (batch_size, 3, rows, cols)

    Output:
        3 Q-values:
            0 = straight
            1 = turn right
            2 = turn left
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_size: int = 3,
        rows: int = 24,
        cols: int = 32,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, rows, cols)
            flattened_size = self.features(dummy_input).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor with shape (batch_size, 3, rows, cols)

        Returns:
            Q-values with shape (batch_size, 3)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def create_cnn_dqn(
    input_channels: int = 3,
    output_size: int = 3,
    rows: int = 24,
    cols: int = 32,
) -> CNNDQN:
    """Factory function for creating a CNNDQN model."""
    return CNNDQN(
        input_channels=input_channels,
        output_size=output_size,
        rows=rows,
        cols=cols,
    )
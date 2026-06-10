from __future__ import annotations

import torch
import torch.nn as nn


class MLPDQN(nn.Module):
    """
    MLP-based Deep Q-Network for Snake.

    Input:
        19-dimensional state vector from SnakeEnv.get_state_vector()

    Output:
        3 Q-values:
            0 = straight
            1 = turn right
            2 = turn left
    """

    def __init__(
        self,
        input_size: int = 19,
        hidden_size: int = 256,
        output_size: int = 3,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor with shape (batch_size, input_size)
               or shape (input_size,)

        Returns:
            Q-values with shape (batch_size, output_size)
            or shape (output_size,) for single input.
        """
        return self.network(x)


def create_mlp_dqn(
    input_size: int = 19,
    hidden_size: int = 256,
    output_size: int = 3,
) -> MLPDQN:
    """Factory function for creating an MLPDQN model."""
    return MLPDQN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
    )
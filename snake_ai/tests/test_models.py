from __future__ import annotations

import os
import sys

import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAKE_AI_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SNAKE_AI_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from snake_ai.src.models.cnn_dqn import CNNDQN
from snake_ai.src.models.mlp_dqn import MLPDQN


def test_mlp_dqn_single_state_output_shape() -> None:
    model = MLPDQN(input_size=19, hidden_size=256, output_size=3)

    state = torch.zeros(19)
    output = model(state)

    assert output.shape == (3,)


def test_mlp_dqn_batch_output_shape() -> None:
    model = MLPDQN(input_size=19, hidden_size=256, output_size=3)

    states = torch.zeros(8, 19)
    output = model(states)

    assert output.shape == (8, 3)


def test_cnn_dqn_batch_output_shape() -> None:
    model = CNNDQN(input_channels=3, output_size=3, rows=24, cols=32)

    states = torch.zeros(8, 3, 24, 32)
    output = model(states)

    assert output.shape == (8, 3)


def test_cnn_dqn_single_grid_needs_batch_dimension() -> None:
    model = CNNDQN(input_channels=3, output_size=3, rows=24, cols=32)

    state = torch.zeros(1, 3, 24, 32)
    output = model(state)

    assert output.shape == (1, 3)
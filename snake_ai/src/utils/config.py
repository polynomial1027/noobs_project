from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SnakeConfig:
    """
    Shared game configuration.

    This config is used by:
        - human play
        - random agent
        - MLP-DQN training
        - CNN-DQN training
    """

    width: int = 640
    height: int = 480
    block_size: int = 20
    speed: int = 30
    max_steps_without_food: int = 100


@dataclass
class MLPDQNConfig:
    """
    Configuration for MLP-DQN.

    State:
        19-dimensional vector from SnakeEnv.get_state_vector()

    Action:
        0 = straight
        1 = turn right
        2 = turn left
    """

    input_size: int = 19
    hidden_size: int = 256
    output_size: int = 3

    learning_rate: float = 0.001
    gamma: float = 0.9
    batch_size: int = 1000
    memory_size: int = 100_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    target_update_every: int = 20

    max_episodes: int = 500
    render_every: int = 50
    save_every: int = 50


@dataclass
class CNNDQNConfig:
    """
    Configuration for CNN-DQN.

    State:
        grid tensor from SnakeEnv.get_state_grid()
        shape = (3, rows, cols)

    Action:
        0 = straight
        1 = turn right
        2 = turn left
    """

    input_channels: int = 3
    output_size: int = 3

    learning_rate: float = 0.0005
    gamma: float = 0.9
    batch_size: int = 256
    memory_size: int = 100_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    target_update_every: int = 20

    max_episodes: int = 500
    render_every: int = 50
    save_every: int = 50
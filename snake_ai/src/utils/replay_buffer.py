from __future__ import annotations

import random
from collections import deque
from typing import Deque, Tuple

import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.

    Stores transitions:
        state, action, reward, next_state, done
    """

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")

        self.capacity = capacity
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Save one transition into the replay buffer."""
        self.memory.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Randomly sample a batch of transitions."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if batch_size > len(self.memory):
            raise ValueError(
                f"Not enough samples in replay buffer. "
                f"Requested {batch_size}, available {len(self.memory)}."
            )

        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.bool_),
        )

    def clear(self) -> None:
        """Remove all stored transitions."""
        self.memory.clear()

    def __len__(self) -> int:
        return len(self.memory)
from __future__ import annotations

import os
import sys

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAKE_AI_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SNAKE_AI_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from snake_ai.src.utils.replay_buffer import ReplayBuffer


def test_replay_buffer_can_add_experience() -> None:
    buffer = ReplayBuffer(capacity=10)

    state = np.zeros(19, dtype=np.float32)
    action = 0
    reward = 1.0
    next_state = np.ones(19, dtype=np.float32)
    done = False

    buffer.push(state, action, reward, next_state, done)

    assert len(buffer) == 1


def test_replay_buffer_respects_capacity() -> None:
    buffer = ReplayBuffer(capacity=3)

    for i in range(5):
        state = np.full(19, i, dtype=np.float32)
        next_state = np.full(19, i + 1, dtype=np.float32)
        buffer.push(state, 0, 1.0, next_state, False)

    assert len(buffer) == 3


def test_replay_buffer_sample_shapes() -> None:
    buffer = ReplayBuffer(capacity=10)

    for i in range(6):
        state = np.full(19, i, dtype=np.float32)
        next_state = np.full(19, i + 1, dtype=np.float32)
        buffer.push(state, i % 3, float(i), next_state, i % 2 == 0)

    states, actions, rewards, next_states, dones = buffer.sample(batch_size=4)

    assert states.shape == (4, 19)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 19)
    assert dones.shape == (4,)


def test_replay_buffer_sample_raises_when_not_enough_data() -> None:
    buffer = ReplayBuffer(capacity=10)

    state = np.zeros(19, dtype=np.float32)
    next_state = np.ones(19, dtype=np.float32)
    buffer.push(state, 0, 1.0, next_state, False)

    try:
        buffer.sample(batch_size=4)
        assert False, "Expected ValueError when sampling more items than available."
    except ValueError:
        assert True
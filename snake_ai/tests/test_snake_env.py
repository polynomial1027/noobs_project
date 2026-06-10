from __future__ import annotations

import os
import sys

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAKE_AI_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SNAKE_AI_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from snake_ai.src.env.snake_env import SnakeEnv


def test_reset_returns_state_vector() -> None:
    env = SnakeEnv(render_mode=False)

    state = env.reset()

    assert isinstance(state, np.ndarray)
    assert state.shape == (19,)
    assert state.dtype == np.float32


def test_step_returns_reward_done_score() -> None:
    env = SnakeEnv(render_mode=False)

    env.reset()
    reward, done, score = env.step(0)

    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(score, int)


def test_state_grid_shape() -> None:
    env = SnakeEnv(render_mode=False)

    env.reset()
    grid = env.get_state_grid()

    assert isinstance(grid, np.ndarray)
    assert grid.shape == (3, env.rows, env.cols)
    assert grid.dtype == np.float32


def test_random_actions_can_run_for_a_while() -> None:
    env = SnakeEnv(render_mode=False)

    env.reset()

    for action in [0, 1, 2, 0, 1, 2, 0, 0, 1, 2]:
        reward, done, score = env.step(action)

        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(score, int)

        if done:
            env.reset()
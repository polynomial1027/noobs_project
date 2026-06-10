from __future__ import annotations

import os
import sys

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAKE_AI_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SNAKE_AI_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from snake_ai.src.env.snake_env import Direction, Point, SnakeEnv


def test_state_vector_has_19_features() -> None:
    env = SnakeEnv(render_mode=False)

    state = env.get_state_vector()

    assert isinstance(state, np.ndarray)
    assert state.shape == (19,)
    assert state.dtype == np.float32


def test_initial_direction_is_right() -> None:
    env = SnakeEnv(render_mode=False)

    state = env.get_state_vector()

    direction_left = state[3]
    direction_right = state[4]
    direction_up = state[5]
    direction_down = state[6]

    assert direction_left == 0
    assert direction_right == 1
    assert direction_up == 0
    assert direction_down == 0


def test_wall_distances_are_normalized() -> None:
    env = SnakeEnv(render_mode=False)

    state = env.get_state_vector()

    wall_distances = state[11:15]

    assert np.all(wall_distances >= 0)
    assert np.all(wall_distances <= 1)


def test_food_distances_are_normalized() -> None:
    env = SnakeEnv(render_mode=False)

    state = env.get_state_vector()

    food_dx = state[15]
    food_dy = state[16]

    assert -1 <= food_dx <= 1
    assert -1 <= food_dy <= 1


def test_collision_detects_wall() -> None:
    env = SnakeEnv(render_mode=False)

    outside_left = Point(-env.block_size, env.head.y)
    outside_right = Point(env.width, env.head.y)
    outside_top = Point(env.head.x, -env.block_size)
    outside_bottom = Point(env.head.x, env.height)

    assert env.is_collision(outside_left)
    assert env.is_collision(outside_right)
    assert env.is_collision(outside_top)
    assert env.is_collision(outside_bottom)


def test_human_reverse_direction_is_blocked() -> None:
    env = SnakeEnv(render_mode=False)

    env.direction = Direction.RIGHT
    env.human_step(Direction.LEFT)

    assert env.direction == Direction.RIGHT
from __future__ import annotations

from typing import Callable, Dict, Literal


RewardMode = Literal[
    "classic",
    "distance",
    "step_penalty",
    "anti_loop",
]


def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    Calculate Manhattan distance between two grid points.
    """
    return abs(x1 - x2) + abs(y1 - y2)


def reward_classic(
    ate_food: bool,
    died: bool,
    old_distance_to_food: int,
    new_distance_to_food: int,
    steps_since_food: int,
    max_steps_without_food: int,
) -> float:
    """
    Original sparse reward.

    eat food: +10
    die: -10
    normal move: 0
    """
    if died:
        return -10.0

    if ate_food:
        return 10.0

    return 0.0


def reward_distance(
    ate_food: bool,
    died: bool,
    old_distance_to_food: int,
    new_distance_to_food: int,
    steps_since_food: int,
    max_steps_without_food: int,
) -> float:
    """
    Distance-shaped reward.

    eat food: +10
    die: -10
    move closer to food: +0.1
    move farther from food: -0.1
    same distance: 0
    """
    if died:
        return -10.0

    if ate_food:
        return 10.0

    if new_distance_to_food < old_distance_to_food:
        return 0.1

    if new_distance_to_food > old_distance_to_food:
        return -0.1

    return 0.0


def reward_step_penalty(
    ate_food: bool,
    died: bool,
    old_distance_to_food: int,
    new_distance_to_food: int,
    steps_since_food: int,
    max_steps_without_food: int,
) -> float:
    """
    Step-penalty reward.

    eat food: +10
    die: -10
    every normal move: -0.01
    move closer to food: +0.1
    move farther from food: -0.1

    This encourages the snake to reach food faster.
    """
    if died:
        return -10.0

    if ate_food:
        return 10.0
    else:
        reward = -0.01

    if new_distance_to_food < old_distance_to_food:
        reward += 0.1
    elif new_distance_to_food > old_distance_to_food:
        reward -= 0.1

    return reward


def reward_anti_loop(
    ate_food: bool,
    died: bool,
    old_distance_to_food: int,
    new_distance_to_food: int,
    steps_since_food: int,
    max_steps_without_food: int,
) -> float:
    """
    Anti-loop reward.

    Based on step_penalty, but adds extra pressure when the snake
    has not eaten for a long time.

    This is designed to reduce useless circling.
    """
    if died:
        return -10.0

    if ate_food:
        return 10.0
    else:
        reward = -0.01

    if new_distance_to_food < old_distance_to_food:
        reward += 0.1
    elif new_distance_to_food > old_distance_to_food:
        reward -= 0.1

    progress_ratio = steps_since_food / max(max_steps_without_food, 1)

    if progress_ratio > 0.5:
        reward -= 0.02

    if progress_ratio > 0.8:
        reward -= 0.05

    return reward


REWARD_FUNCTIONS: Dict[str, Callable[..., float]] = {
    "classic": reward_classic,
    "distance": reward_distance,
    "step_penalty": reward_step_penalty,
    "anti_loop": reward_anti_loop,
}


def calculate_reward(
    reward_mode: str,
    ate_food: bool,
    died: bool,
    old_distance_to_food: int,
    new_distance_to_food: int,
    steps_since_food: int,
    max_steps_without_food: int,
) -> float:
    """
    Calculate reward by selected reward mode.
    """
    if reward_mode not in REWARD_FUNCTIONS:
        available_modes = ", ".join(REWARD_FUNCTIONS.keys())
        raise ValueError(
            f"Unknown reward mode: {reward_mode}. "
            f"Available reward modes: {available_modes}"
        )

    reward_function = REWARD_FUNCTIONS[reward_mode]

    return reward_function(
        ate_food=ate_food,
        died=died,
        old_distance_to_food=old_distance_to_food,
        new_distance_to_food=new_distance_to_food,
        steps_since_food=steps_since_food,
        max_steps_without_food=max_steps_without_food,
    )


def get_available_reward_modes() -> list[str]:
    """
    Return all available reward modes.
    """
    return list(REWARD_FUNCTIONS.keys())
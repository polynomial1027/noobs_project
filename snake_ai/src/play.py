from __future__ import annotations

import argparse
import os
import sys
import time

import pygame


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAKE_AI_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SNAKE_AI_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from snake_ai.src.agents.cnn_dqn_agent import CNNDQNAgent
from snake_ai.src.agents.mlp_dqn_agent import MLPDQNAgent
from snake_ai.src.agents.random_agent import RandomAgent
from snake_ai.src.env.snake_env import Direction, SnakeEnv
from snake_ai.src.utils.config import CNNDQNConfig, MLPDQNConfig, SnakeConfig


def handle_quit_events() -> bool:
    """
    Return True if the user wants to quit.
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True

    return False


def play_human() -> None:
    snake_config = SnakeConfig()

    env = SnakeEnv(
        width=snake_config.width,
        height=snake_config.height,
        block_size=snake_config.block_size,
        speed=12,
        render_mode=True,
        max_steps_without_food=500,
    )

    running = True
    current_direction = Direction.RIGHT

    print("Human mode started.")
    print("Use arrow keys or WASD to control the snake.")
    print("Press ESC to quit.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in [pygame.K_LEFT, pygame.K_a]:
                    current_direction = Direction.LEFT
                elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                    current_direction = Direction.RIGHT
                elif event.key in [pygame.K_UP, pygame.K_w]:
                    current_direction = Direction.UP
                elif event.key in [pygame.K_DOWN, pygame.K_s]:
                    current_direction = Direction.DOWN

        reward, done, score = env.human_step(current_direction)

        if done:
            print(f"Game over. Score: {score}, Reward: {reward}")
            time.sleep(0.8)
            env.reset()
            current_direction = Direction.RIGHT

    env.close()


def play_random() -> None:
    snake_config = SnakeConfig()

    env = SnakeEnv(
        width=snake_config.width,
        height=snake_config.height,
        block_size=snake_config.block_size,
        speed=20,
        render_mode=True,
        max_steps_without_food=snake_config.max_steps_without_food,
    )

    agent = RandomAgent(action_size=3)

    print("Random agent mode started.")
    print("Press ESC or close the window to quit.")

    running = True

    while running:
        if handle_quit_events():
            running = False
            break

        state = env.get_state_vector()
        action = agent.get_action(state)

        reward, done, score = env.step(action)

        if done:
            print(f"Random agent died. Score: {score}, Reward: {reward}")
            time.sleep(0.3)
            env.reset()

    env.close()


def play_mlp_dqn(checkpoint_path: str) -> None:
    snake_config = SnakeConfig()
    dqn_config = MLPDQNConfig()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train first with: python snake_ai/src/train_mlp_dqn.py"
        )

    env = SnakeEnv(
        width=snake_config.width,
        height=snake_config.height,
        block_size=snake_config.block_size,
        speed=snake_config.speed,
        render_mode=True,
        max_steps_without_food=snake_config.max_steps_without_food,
    )

    agent = MLPDQNAgent(
        input_size=dqn_config.input_size,
        hidden_size=dqn_config.hidden_size,
        output_size=dqn_config.output_size,
        learning_rate=dqn_config.learning_rate,
        gamma=dqn_config.gamma,
        batch_size=dqn_config.batch_size,
        memory_size=dqn_config.memory_size,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
        target_update_every=dqn_config.target_update_every,
    )

    agent.load(checkpoint_path)
    agent.epsilon = 0.0

    print("MLP-DQN play mode started.")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Device: {agent.device}")
    print("Press ESC or close the window to quit.")

    running = True
    env.reset()

    while running:
        if handle_quit_events():
            running = False
            break

        state = env.get_state_vector()
        action = agent.get_action(state, training=False)

        reward, done, score = env.step(action)

        if done:
            print(f"MLP-DQN died. Score: {score}, Reward: {reward}")
            time.sleep(0.5)
            env.reset()

    env.close()


def play_cnn_dqn(checkpoint_path: str) -> None:
    snake_config = SnakeConfig()
    dqn_config = CNNDQNConfig()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train first with: python snake_ai/src/train_cnn_dqn.py"
        )

    env = SnakeEnv(
        width=snake_config.width,
        height=snake_config.height,
        block_size=snake_config.block_size,
        speed=snake_config.speed,
        render_mode=True,
        max_steps_without_food=snake_config.max_steps_without_food,
    )

    agent = CNNDQNAgent(
        input_channels=dqn_config.input_channels,
        output_size=dqn_config.output_size,
        rows=env.rows,
        cols=env.cols,
        learning_rate=dqn_config.learning_rate,
        gamma=dqn_config.gamma,
        batch_size=dqn_config.batch_size,
        memory_size=dqn_config.memory_size,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
        target_update_every=dqn_config.target_update_every,
    )

    agent.load(checkpoint_path)
    agent.epsilon = 0.0

    print("CNN-DQN play mode started.")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Device: {agent.device}")
    print("Press ESC or close the window to quit.")

    running = True
    env.reset()

    while running:
        if handle_quit_events():
            running = False
            break

        state = env.get_state_grid()
        action = agent.get_action(state, training=False)

        reward, done, score = env.step(action)

        if done:
            print(f"CNN-DQN died. Score: {score}, Reward: {reward}")
            time.sleep(0.5)
            env.reset()

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Snake manually or with an agent.")
    parser.add_argument(
        "--mode",
        type=str,
        default="human",
        choices=["human", "random", "mlp", "cnn"],
        help="Choose play mode: human, random, mlp, or cnn.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint for mlp or cnn mode.",
    )

    args = parser.parse_args()

    if args.mode == "human":
        play_human()
    elif args.mode == "random":
        play_random()
    elif args.mode == "mlp":
        checkpoint_path = args.checkpoint or "snake_ai/checkpoints/mlp_dqn_latest.pth"
        play_mlp_dqn(checkpoint_path)
    elif args.mode == "cnn":
        checkpoint_path = args.checkpoint or "snake_ai/checkpoints/cnn_dqn_latest.pth"
        play_cnn_dqn(checkpoint_path)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
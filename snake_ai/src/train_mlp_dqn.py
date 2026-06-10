from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAKE_AI_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SNAKE_AI_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from snake_ai.src.agents.mlp_dqn_agent import MLPDQNAgent
from snake_ai.src.env.snake_env import SnakeEnv
from snake_ai.src.rewards.reward_functions import get_available_reward_modes
from snake_ai.src.utils.config import MLPDQNConfig, SnakeConfig
from snake_ai.src.utils.timer import TrainingTimer


def train_mlp_dqn(
    max_episodes: int,
    render_every: int,
    save_every: int,
    checkpoint_path: str,
    reward_mode: str,
    resume_path: str | None = None,
) -> None:
    snake_config = SnakeConfig()
    dqn_config = MLPDQNConfig()

    scores_window = deque(maxlen=100)
    record_score = 0

    timer = TrainingTimer()

    env = SnakeEnv(
        width=snake_config.width,
        height=snake_config.height,
        block_size=snake_config.block_size,
        speed=snake_config.speed,
        render_mode=False,
        max_steps_without_food=snake_config.max_steps_without_food,
        reward_mode=reward_mode,
    )

    agent = MLPDQNAgent(
        input_size=dqn_config.input_size,
        hidden_size=dqn_config.hidden_size,
        output_size=dqn_config.output_size,
        learning_rate=dqn_config.learning_rate,
        gamma=dqn_config.gamma,
        batch_size=dqn_config.batch_size,
        memory_size=dqn_config.memory_size,
        epsilon_start=dqn_config.epsilon_start,
        epsilon_end=dqn_config.epsilon_end,
        epsilon_decay=dqn_config.epsilon_decay,
        target_update_every=dqn_config.target_update_every,
    )

    if resume_path is not None:
        agent.load(resume_path)
        print(f"Loaded checkpoint from: {resume_path}")

    print("MLP-DQN training started.")
    print(f"Device: {agent.device}")
    print(f"State size: {dqn_config.input_size}")
    print(f"Reward mode: {reward_mode}")
    print(f"Max episodes: {max_episodes}")
    print(f"Render every: {render_every}")
    print("-" * 80)

    for episode in range(1, max_episodes + 1):
        should_render = render_every > 0 and episode % render_every == 0

        env.close()
        env = SnakeEnv(
            width=snake_config.width,
            height=snake_config.height,
            block_size=snake_config.block_size,
            speed=snake_config.speed,
            render_mode=should_render,
            max_steps_without_food=snake_config.max_steps_without_food,
            reward_mode=reward_mode,
        )

        state = env.reset()

        done = False
        score = 0
        episode_loss_values = []

        timer.start_episode()

        while not done:
            action = agent.get_action(state, training=True)

            reward, done, score = env.step(action)
            next_state = env.get_state_vector()

            agent.remember(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                episode_loss_values.append(loss)

            state = next_state

        agent.end_episode()

        scores_window.append(score)
        record_score = max(record_score, score)

        timing = timer.end_episode(episode)

        avg_score = float(np.mean(scores_window)) if scores_window else 0.0
        avg_loss = float(np.mean(episode_loss_values)) if episode_loss_values else 0.0

        print(
            f"Episode: {episode:4d} | "
            f"Score: {score:3d} | "
            f"Record: {record_score:3d} | "
            f"Avg100: {avg_score:6.2f} | "
            f"Loss: {avg_loss:8.5f} | "
            f"Epsilon: {agent.epsilon:6.3f} | "
            f"Episode time: {timing.elapsed_seconds:6.2f}s | "
            f"Total: {TrainingTimer.format_seconds(timing.total_elapsed_seconds)}"
        )

        if save_every > 0 and episode % save_every == 0:
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to: {checkpoint_path}")

    agent.save(checkpoint_path)
    print(f"Final checkpoint saved to: {checkpoint_path}")

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP-DQN agent on Snake.")

    parser.add_argument(
        "--episodes",
        type=int,
        default=MLPDQNConfig.max_episodes,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=MLPDQNConfig.render_every,
        help="Render one episode every N episodes. Use 0 to disable rendering.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=MLPDQNConfig.save_every,
        help="Save checkpoint every N episodes. Use 0 to disable periodic saving.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="snake_ai/checkpoints/mlp_dqn_latest.pth",
        help="Path to save checkpoint.",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="classic",
        choices=get_available_reward_modes(),
        help="Reward function mode.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )

    args = parser.parse_args()

    train_mlp_dqn(
        max_episodes=args.episodes,
        render_every=args.render_every,
        save_every=args.save_every,
        checkpoint_path=args.checkpoint,
        reward_mode=args.reward,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
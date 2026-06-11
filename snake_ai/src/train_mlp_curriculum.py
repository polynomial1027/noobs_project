from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from typing import Sequence

import matplotlib.pyplot as plt
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
from snake_ai.src.utils.plotting import save_training_log_csv
from snake_ai.src.utils.timer import TrainingTimer


DEFAULT_REWARD_SEQUENCE = [
    "classic",
    "distance",
    "step_penalty",
    "anti_loop",
]


def plot_curriculum_summary(
    rows: list[dict],
    path: str,
    title: str = "MLP-DQN curriculum training summary",
) -> None:
    """
    Plot score, Avg100, loss, and reward stage in one image.
    """
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    global_episodes = [row["global_episode"] for row in rows]
    scores = [row["score"] for row in rows]
    avg_scores = [row["avg100"] for row in rows]
    losses = [row["loss"] for row in rows]
    stage_indices = [row["stage"] for row in rows]

    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.plot(global_episodes, scores)
    plt.xlabel("Global episode")
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(global_episodes, avg_scores)
    plt.xlabel("Global episode")
    plt.ylabel("Avg100 score")
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(global_episodes, losses)
    plt.xlabel("Global episode")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(global_episodes, stage_indices)
    plt.xlabel("Global episode")
    plt.ylabel("Stage")
    plt.yticks(sorted(set(stage_indices)))
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def train_one_curriculum_stage(
    agent: MLPDQNAgent,
    reward_mode: str,
    stage_index: int,
    stage_episodes: int,
    global_episode_start: int,
    render_every: int,
    save_every: int,
    checkpoint_path: str,
    stage_csv_path: str,
) -> tuple[int, list[dict]]:
    """
    Train one curriculum stage using the provided reward mode.

    Args:
        agent:
            The same MLPDQNAgent instance across all stages.
        reward_mode:
            Reward function used in this stage.
        stage_index:
            Stage number starting from 1.
        stage_episodes:
            Number of episodes for this stage.
        global_episode_start:
            Global episode number before this stage starts.
        render_every:
            Render one episode every N global episodes. 0 disables rendering.
        save_every:
            Save checkpoint every N stage episodes. 0 disables periodic saving.
        checkpoint_path:
            Checkpoint path for this stage.
        stage_csv_path:
            Per-stage CSV path.

    Returns:
        updated global episode number and stage rows.
    """
    snake_config = SnakeConfig()

    scores_window = deque(maxlen=100)
    record_score = 0

    timer = TrainingTimer()
    stage_rows: list[dict] = []

    env = SnakeEnv(
        width=snake_config.width,
        height=snake_config.height,
        block_size=snake_config.block_size,
        speed=snake_config.speed,
        render_mode=False,
        max_steps_without_food=snake_config.max_steps_without_food,
        reward_mode=reward_mode,
    )

    print()
    print("=" * 90)
    print(f"Stage {stage_index} started.")
    print(f"Reward mode: {reward_mode}")
    print(f"Stage episodes: {stage_episodes}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 90)

    global_episode = global_episode_start

    for stage_episode in range(1, stage_episodes + 1):
        global_episode += 1

        should_render = render_every > 0 and global_episode % render_every == 0

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

        timing = timer.end_episode(stage_episode)

        avg_score = float(np.mean(scores_window)) if scores_window else 0.0
        avg_loss = float(np.mean(episode_loss_values)) if episode_loss_values else 0.0

        row = {
            "global_episode": global_episode,
            "stage": stage_index,
            "stage_episode": stage_episode,
            "reward_mode": reward_mode,
            "score": score,
            "stage_record": record_score,
            "avg100": avg_score,
            "loss": avg_loss,
            "epsilon": agent.epsilon,
            "episode_time": timing.elapsed_seconds,
            "stage_total_time": timing.total_elapsed_seconds,
            "training_steps": agent.training_steps,
            "agent_episode_count": agent.episode_count,
        }

        stage_rows.append(row)

        print(
            f"Global: {global_episode:5d} | "
            f"Stage: {stage_index} | "
            f"Stage Ep: {stage_episode:4d} | "
            f"Reward: {reward_mode:12s} | "
            f"Score: {score:3d} | "
            f"Record: {record_score:3d} | "
            f"Avg100: {avg_score:6.2f} | "
            f"Loss: {avg_loss:8.5f} | "
            f"Epsilon: {agent.epsilon:6.3f} | "
            f"Episode time: {timing.elapsed_seconds:6.2f}s | "
            f"Stage total: {TrainingTimer.format_seconds(timing.total_elapsed_seconds)}"
        )

        if save_every > 0 and stage_episode % save_every == 0:
            agent.save(checkpoint_path)
            print(f"Saved stage checkpoint to: {checkpoint_path}")

    agent.save(checkpoint_path)
    print(f"Final stage checkpoint saved to: {checkpoint_path}")

    save_training_log_csv(
        path=stage_csv_path,
        rows=stage_rows,
    )
    print(f"Stage CSV saved to: {stage_csv_path}")

    env.close()

    return global_episode, stage_rows


def train_mlp_curriculum(
    stage_episodes: int,
    reward_sequence: Sequence[str],
    render_every: int,
    save_every: int,
    output_prefix: str,
    resume_path: str | None = None,
) -> None:
    """
    Train MLP-DQN through a reward curriculum.

    Example sequence:
        classic -> distance -> step_penalty -> anti_loop
    """
    available_modes = set(get_available_reward_modes())

    for reward_mode in reward_sequence:
        if reward_mode not in available_modes:
            raise ValueError(
                f"Unknown reward mode: {reward_mode}. "
                f"Available modes: {', '.join(sorted(available_modes))}"
            )

    dqn_config = MLPDQNConfig()

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

    print("MLP-DQN curriculum training started.")
    print(f"Device: {agent.device}")
    print(f"State size: {dqn_config.input_size}")
    print(f"Reward sequence: {' -> '.join(reward_sequence)}")
    print(f"Episodes per stage: {stage_episodes}")
    print(f"Render every: {render_every}")
    print(f"Save every: {save_every}")
    print(f"Output prefix: {output_prefix}")

    all_rows: list[dict] = []
    global_episode = 0

    for stage_index, reward_mode in enumerate(reward_sequence, start=1):
        stage_checkpoint_path = (
            f"snake_ai/checkpoints/{output_prefix}_stage_{stage_index}_{reward_mode}.pth"
        )
        stage_csv_path = (
            f"snake_ai/runs/{output_prefix}_{reward_mode}_training.csv"
        )

        global_episode, stage_rows = train_one_curriculum_stage(
            agent=agent,
            reward_mode=reward_mode,
            stage_index=stage_index,
            stage_episodes=stage_episodes,
            global_episode_start=global_episode,
            render_every=render_every,
            save_every=save_every,
            checkpoint_path=stage_checkpoint_path,
            stage_csv_path=stage_csv_path,
        )

        all_rows.extend(stage_rows)

    final_checkpoint_path = f"snake_ai/checkpoints/{output_prefix}_final.pth"
    combined_csv_path = f"snake_ai/runs/{output_prefix}_training.csv"
    combined_plot_path = f"snake_ai/runs/{output_prefix}_training.png"

    agent.save(final_checkpoint_path)
    print(f"Final curriculum checkpoint saved to: {final_checkpoint_path}")

    save_training_log_csv(
        path=combined_csv_path,
        rows=all_rows,
    )
    print(f"Combined CSV saved to: {combined_csv_path}")

    plot_curriculum_summary(
        rows=all_rows,
        path=combined_plot_path,
        title=f"MLP-DQN curriculum training: {' -> '.join(reward_sequence)}",
    )
    print(f"Combined plot saved to: {combined_plot_path}")

    print()
    print("Curriculum training finished.")
    print(f"Final model: {final_checkpoint_path}")
    print(f"Full data: {combined_csv_path}")
    print(f"Plot: {combined_plot_path}")


def parse_reward_sequence(raw_sequence: str) -> list[str]:
    """
    Parse comma-separated reward sequence.

    Example:
        classic,distance,step_penalty,anti_loop
    """
    return [item.strip() for item in raw_sequence.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MLP-DQN using a staged reward curriculum."
    )

    parser.add_argument(
        "--stage-episodes",
        type=int,
        default=300,
        help="Number of episodes per reward stage.",
    )
    parser.add_argument(
        "--reward-sequence",
        type=str,
        default="classic,distance,step_penalty,anti_loop",
        help="Comma-separated reward sequence.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=0,
        help="Render one episode every N global episodes. Use 0 to disable rendering.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save checkpoint every N stage episodes. Use 0 to disable periodic saving.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="mlp_curriculum",
        help="Prefix for checkpoints, CSV files, and plot files.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint to resume curriculum training from.",
    )

    args = parser.parse_args()

    reward_sequence = parse_reward_sequence(args.reward_sequence)

    train_mlp_curriculum(
        stage_episodes=args.stage_episodes,
        reward_sequence=reward_sequence,
        render_every=args.render_every,
        save_every=args.save_every,
        output_prefix=args.output_prefix,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
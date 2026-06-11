from __future__ import annotations

import csv
import os
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


def save_training_log_csv(
    path: str,
    rows: Iterable[dict],
    fieldnames: Sequence[str] | None = None,
) -> None:
    """
    Save training records to a CSV file.

    Example row:
        {
            "episode": 1,
            "score": 0,
            "record": 0,
            "avg100": 0.0,
            "loss": 0.0,
            "epsilon": 0.995,
            "episode_time": 0.2,
            "total_time": 0.2,
            "reward_mode": "classic",
        }
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    rows = list(rows)

    if not rows:
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_scores(
    scores: Sequence[int | float],
    path: str | None = None,
    title: str = "Snake AI training scores",
) -> None:
    """
    Plot episode scores.
    """
    if not scores:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(True)

    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()


def plot_avg_scores(
    avg_scores: Sequence[int | float],
    path: str | None = None,
    title: str = "Snake AI average score",
) -> None:
    """
    Plot moving average scores.
    """
    if not avg_scores:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(avg_scores)
    plt.xlabel("Episode")
    plt.ylabel("Average score")
    plt.title(title)
    plt.grid(True)

    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()


def plot_losses(
    losses: Sequence[int | float],
    path: str | None = None,
    title: str = "Snake AI training loss",
) -> None:
    """
    Plot training loss.
    """
    if not losses:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)

    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()


def plot_training_summary(
    scores: Sequence[int | float],
    avg_scores: Sequence[int | float],
    losses: Sequence[int | float],
    path: str,
    title: str = "Snake AI training summary",
) -> None:
    """
    Plot score, Avg100 score, and loss in one image.

    Output:
        A PNG image with three charts:
            1. Episode score
            2. Moving average score
            3. Training loss
    """
    if not scores:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    episodes = list(range(1, len(scores) + 1))

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(episodes, scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(episodes, avg_scores)
    plt.xlabel("Episode")
    plt.ylabel("Avg100 score")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(episodes, losses)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
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

    Args:
        scores: score per episode
        path: optional path to save image
        title: plot title
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
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class EpisodeTiming:
    """
    Timing result for one episode.
    """

    episode: int
    elapsed_seconds: float
    total_elapsed_seconds: float


class TrainingTimer:
    """
    Simple timer for tracking training speed.

    It measures:
        - total training time
        - time spent in each episode
    """

    def __init__(self) -> None:
        self.training_start_time = time.time()
        self.episode_start_time = time.time()

    def start_episode(self) -> None:
        """Mark the beginning of a new episode."""
        self.episode_start_time = time.time()

    def end_episode(self, episode: int) -> EpisodeTiming:
        """Return timing information for the finished episode."""
        now = time.time()

        return EpisodeTiming(
            episode=episode,
            elapsed_seconds=now - self.episode_start_time,
            total_elapsed_seconds=now - self.training_start_time,
        )

    @staticmethod
    def format_seconds(seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        seconds = int(seconds)

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60

        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
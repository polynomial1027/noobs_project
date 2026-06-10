from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from snake_ai.src.models.cnn_dqn import CNNDQN
from snake_ai.src.utils.replay_buffer import ReplayBuffer


class CNNDQNAgent:
    """
    DQN agent using CNN grid state.

    State:
        SnakeEnv.get_state_grid()
        shape = (3, rows, cols)

    Action space:
        0 = straight
        1 = turn right
        2 = turn left
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_size: int = 3,
        rows: int = 24,
        cols: int = 32,
        learning_rate: float = 0.0005,
        gamma: float = 0.9,
        batch_size: int = 256,
        memory_size: int = 100_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_every: int = 20,
        device: str | None = None,
    ) -> None:
        self.input_channels = input_channels
        self.output_size = output_size
        self.rows = rows
        self.cols = cols

        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_every = target_update_every

        self.device = torch.device(device or self._get_default_device())

        self.policy_net = CNNDQN(
            input_channels=input_channels,
            output_size=output_size,
            rows=rows,
            cols=cols,
        ).to(self.device)

        self.target_net = CNNDQN(
            input_channels=input_channels,
            output_size=output_size,
            rows=rows,
            cols=cols,
        ).to(self.device)

        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.memory = ReplayBuffer(capacity=memory_size)

        self.training_steps = 0
        self.episode_count = 0

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose an action using epsilon-greedy.

        Args:
            state: numpy array with shape (3, rows, cols)
            training: whether to use random exploration

        Returns:
            action index
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.output_size - 1)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        self.policy_net.train()

        return int(action)

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """
        Train the policy network using one random batch from replay memory.

        Returns:
            loss value if trained
            None if there is not enough replay memory yet
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)

        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_tensor).max(dim=1)[0]
            target_q_values = rewards_tensor + self.gamma * next_q_values * (~dones_tensor)

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1

        return float(loss.item())

    def end_episode(self) -> None:
        """Update epsilon and target network after one episode."""
        self.episode_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if self.episode_count % self.target_update_every == 0:
            self.update_target_network()

    def update_target_network(self) -> None:
        """Copy policy network parameters to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        """Save model and training metadata."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "episode_count": self.episode_count,
            "rows": self.rows,
            "cols": self.cols,
        }

        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load model and training metadata."""
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.training_steps = checkpoint.get("training_steps", 0)
        self.episode_count = checkpoint.get("episode_count", 0)

        self.policy_net.train()
        self.target_net.eval()

    @staticmethod
    def _get_default_device() -> str:
        """
        Choose the best available device.

        Windows NVIDIA:
            cuda

        Apple Silicon Mac:
            mps

        Fallback:
            cpu
        """
        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"
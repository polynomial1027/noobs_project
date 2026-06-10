from __future__ import annotations

import random


class RandomAgent:
    """
    Random baseline agent for SnakeEnv.

    Action space:
        0 = straight
        1 = turn right
        2 = turn left

    This agent is mainly used to test whether the environment works.
    It does not learn.
    """

    def __init__(self, action_size: int = 3) -> None:
        self.action_size = action_size

    def get_action(self, state=None) -> int:
        """
        Return a random valid action.

        The state argument is accepted for compatibility with future agents,
        but RandomAgent does not use it.
        """
        return random.randint(0, self.action_size - 1)
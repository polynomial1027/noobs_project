from __future__ import annotations

import random
from collections import namedtuple
from enum import Enum
from typing import List, Tuple

import numpy as np
import pygame


Point = namedtuple("Point", "x y")


class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3


class SnakeEnv:
    """
    Pygame Snake environment for both human play and AI training.

    Action space:
        0 = straight
        1 = turn right
        2 = turn left

    State types:
        get_state_vector() -> MLP-DQN input
        get_state_grid()   -> CNN-DQN input
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        block_size: int = 20,
        speed: int = 15,
        render_mode: bool = True,
        max_steps_without_food: int = 100,
    ) -> None:
        if width % block_size != 0 or height % block_size != 0:
            raise ValueError("width and height must be divisible by block_size.")

        self.width = width
        self.height = height
        self.block_size = block_size
        self.speed = speed
        self.render_mode = render_mode
        self.max_steps_without_food = max_steps_without_food

        self.cols = self.width // self.block_size
        self.rows = self.height // self.block_size

        self.display = None
        self.clock = None

        if self.render_mode:
            pygame.init()
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake AI Environment")
            self.clock = pygame.time.Clock()

        self.direction: Direction = Direction.RIGHT
        self.head: Point = Point(self.width // 2, self.height // 2)
        self.snake: List[Point] = []
        self.food: Point = Point(0, 0)
        self.score: int = 0
        self.frame_iteration: int = 0
        self.steps_since_food: int = 0

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the game and return the initial vector state."""
        self.direction = Direction.RIGHT

        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - self.block_size, self.head.y),
            Point(self.head.x - 2 * self.block_size, self.head.y),
        ]

        self.score = 0
        self.frame_iteration = 0
        self.steps_since_food = 0

        self._place_food()

        if self.render_mode:
            self.render()

        return self.get_state_vector()

    def step(self, action: int) -> Tuple[float, bool, int]:
        """
        Execute one action.

        Returns:
            reward: float
            done: bool
            score: int
        """
        self.frame_iteration += 1
        self.steps_since_food += 1

        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0.0
        done = False

        if self.is_collision() or self.steps_since_food > self.max_steps_without_food:
            done = True
            reward = -10.0
            return reward, done, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10.0
            self.steps_since_food = 0
            self._place_food()
        else:
            self.snake.pop()

        if self.render_mode:
            self.render()

        return reward, done, self.score

    def human_step(self, direction: Direction) -> Tuple[float, bool, int]:
        """
        Execute one human direction command.

        This is for keyboard play. It uses absolute direction commands
        but blocks illegal reverse movement.
        """
        if not self._is_reverse_direction(direction):
            self.direction = direction

        self.frame_iteration += 1
        self.steps_since_food += 1

        self._move_forward()
        self.snake.insert(0, self.head)

        reward = 0.0
        done = False

        if self.is_collision() or self.steps_since_food > self.max_steps_without_food:
            done = True
            reward = -10.0
            return reward, done, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10.0
            self.steps_since_food = 0
            self._place_food()
        else:
            self.snake.pop()

        if self.render_mode:
            self.render()

        return reward, done, self.score

    def render(self) -> None:
        """Draw the current game state."""
        if not self.render_mode or self.display is None or self.clock is None:
            return

        background = (20, 20, 20)
        snake_head_color = (30, 180, 90)
        snake_body_color = (40, 130, 80)
        food_color = (220, 60, 60)
        grid_color = (35, 35, 35)
        text_color = (230, 230, 230)

        self.display.fill(background)

        for x in range(0, self.width, self.block_size):
            pygame.draw.line(self.display, grid_color, (x, 0), (x, self.height))

        for y in range(0, self.height, self.block_size):
            pygame.draw.line(self.display, grid_color, (0, y), (self.width, y))

        for index, point in enumerate(self.snake):
            rect = pygame.Rect(point.x, point.y, self.block_size, self.block_size)
            color = snake_head_color if index == 0 else snake_body_color
            pygame.draw.rect(self.display, color, rect)
            pygame.draw.rect(self.display, background, rect, 1)

        food_rect = pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size)
        pygame.draw.rect(self.display, food_color, food_rect)

        font = pygame.font.SysFont("arial", 24)
        text = font.render(
            f"Score: {self.score} | Steps: {self.frame_iteration} | Since food: {self.steps_since_food}",
            True,
            text_color,
        )
        self.display.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.speed)

    def close(self) -> None:
        """Close pygame resources."""
        if self.render_mode:
            pygame.quit()

    def is_collision(self, point: Point | None = None) -> bool:
        """Check wall or self collision."""
        if point is None:
            point = self.head

        hit_left_wall = point.x < 0
        hit_right_wall = point.x >= self.width
        hit_top_wall = point.y < 0
        hit_bottom_wall = point.y >= self.height

        if hit_left_wall or hit_right_wall or hit_top_wall or hit_bottom_wall:
            return True

        if point in self.snake[1:]:
            return True

        return False

    def get_state_vector(self) -> np.ndarray:
        """
        Return enhanced MLP state vector.

        Features:
            0-2: danger straight, right, left
            3-6: current direction one-hot
            7-10: food relative position
            11-14: normalized wall distances
            15-16: normalized food dx, dy
            17: normalized snake length
            18: normalized steps since food
        """
        point_left = Point(self.head.x - self.block_size, self.head.y)
        point_right = Point(self.head.x + self.block_size, self.head.y)
        point_up = Point(self.head.x, self.head.y - self.block_size)
        point_down = Point(self.head.x, self.head.y + self.block_size)

        dir_left = self.direction == Direction.LEFT
        dir_right = self.direction == Direction.RIGHT
        dir_up = self.direction == Direction.UP
        dir_down = self.direction == Direction.DOWN

        danger_straight = (
            (dir_right and self.is_collision(point_right))
            or (dir_left and self.is_collision(point_left))
            or (dir_up and self.is_collision(point_up))
            or (dir_down and self.is_collision(point_down))
        )

        danger_right = (
            (dir_up and self.is_collision(point_right))
            or (dir_down and self.is_collision(point_left))
            or (dir_left and self.is_collision(point_up))
            or (dir_right and self.is_collision(point_down))
        )

        danger_left = (
            (dir_down and self.is_collision(point_right))
            or (dir_up and self.is_collision(point_left))
            or (dir_right and self.is_collision(point_up))
            or (dir_left and self.is_collision(point_down))
        )

        wall_distance_left = self.head.x / max(self.width - self.block_size, 1)
        wall_distance_right = (self.width - self.block_size - self.head.x) / max(
            self.width - self.block_size, 1
        )
        wall_distance_up = self.head.y / max(self.height - self.block_size, 1)
        wall_distance_down = (self.height - self.block_size - self.head.y) / max(
            self.height - self.block_size, 1
        )

        food_dx = (self.food.x - self.head.x) / max(self.width - self.block_size, 1)
        food_dy = (self.food.y - self.head.y) / max(self.height - self.block_size, 1)

        max_snake_length = self.rows * self.cols
        snake_length_normalized = len(self.snake) / max_snake_length
        steps_since_food_normalized = self.steps_since_food / self.max_steps_without_food

        state = np.array(
            [
                int(danger_straight),
                int(danger_right),
                int(danger_left),
                int(dir_left),
                int(dir_right),
                int(dir_up),
                int(dir_down),
                int(self.food.x < self.head.x),
                int(self.food.x > self.head.x),
                int(self.food.y < self.head.y),
                int(self.food.y > self.head.y),
                wall_distance_left,
                wall_distance_right,
                wall_distance_up,
                wall_distance_down,
                food_dx,
                food_dy,
                snake_length_normalized,
                steps_since_food_normalized,
            ],
            dtype=np.float32,
        )

        return state

    def get_state_grid(self) -> np.ndarray:
        """
        Return CNN state grid with shape (3, rows, cols).

        Channel 0: snake head
        Channel 1: snake body
        Channel 2: food
        """
        grid = np.zeros((3, self.rows, self.cols), dtype=np.float32)

        head_col = self.head.x // self.block_size
        head_row = self.head.y // self.block_size
        grid[0, head_row, head_col] = 1.0

        for body_part in self.snake[1:]:
            col = body_part.x // self.block_size
            row = body_part.y // self.block_size
            grid[1, row, col] = 1.0

        food_col = self.food.x // self.block_size
        food_row = self.food.y // self.block_size
        grid[2, food_row, food_col] = 1.0

        return grid

    def _place_food(self) -> None:
        """Randomly place food outside the snake body."""
        while True:
            x = random.randint(0, self.cols - 1) * self.block_size
            y = random.randint(0, self.rows - 1) * self.block_size
            self.food = Point(x, y)

            if self.food not in self.snake:
                break

    def _move(self, action: int) -> None:
        """Move snake using relative action."""
        if action not in [0, 1, 2]:
            raise ValueError("action must be 0, 1, or 2.")

        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_index = clockwise.index(self.direction)

        if action == 0:
            new_direction = clockwise[current_index]
        elif action == 1:
            new_direction = clockwise[(current_index + 1) % 4]
        else:
            new_direction = clockwise[(current_index - 1) % 4]

        self.direction = new_direction
        self._move_forward()

    def _move_forward(self) -> None:
        """Move one block in the current direction."""
        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size

        self.head = Point(x, y)

    def _is_reverse_direction(self, new_direction: Direction) -> bool:
        """Check whether the requested human direction is illegal reverse movement."""
        return (
            (self.direction == Direction.RIGHT and new_direction == Direction.LEFT)
            or (self.direction == Direction.LEFT and new_direction == Direction.RIGHT)
            or (self.direction == Direction.UP and new_direction == Direction.DOWN)
            or (self.direction == Direction.DOWN and new_direction == Direction.UP)
        )
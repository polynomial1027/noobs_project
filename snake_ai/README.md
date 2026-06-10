# Snake AI
This project explores how to train AI agents to play Snake.
The goal is to build a reusable Snake game environment with pygame, then test different algorithms on the same environment.
## Current algorithms
- Human player
- Random agent
- MLP-DQN
- CNN-DQN
## Project structure
```text
snake_ai/
├── README.md
├── requirements.txt
├── src/
│   ├── play.py
│   ├── train_mlp_dqn.py
│   ├── train_cnn_dqn.py
│   ├── env/
│   │   └── snake_env.py
│   ├── agents/
│   │   ├── random_agent.py
│   │   ├── mlp_dqn_agent.py
│   │   └── cnn_dqn_agent.py
│   ├── models/
│   │   ├── mlp_dqn.py
│   │   └── cnn_dqn.py
│   └── utils/
│       ├── config.py
│       ├── replay_buffer.py
│       ├── plotting.py
│       └── timer.py
├── tests/
├── checkpoints/
├── runs/
├── assets/
└── docs/
```
Environment

The game environment is implemented in:

snake_ai/src/env/snake_env.py

The environment supports:

* Human play with keyboard controls
* Random agent play
* MLP-DQN training with vector state
* CNN-DQN training with grid state
* Optional rendering during training

Action space

The environment uses relative actions:

0 = go straight
1 = turn right
2 = turn left

Using relative actions avoids illegal direct reverse movement.

MLP-DQN state

The MLP-DQN agent uses a 19-dimensional state vector:

danger_straight
danger_right
danger_left
direction_left
direction_right
direction_up
direction_down
food_left
food_right
food_up
food_down
wall_distance_left
wall_distance_right
wall_distance_up
wall_distance_down
food_distance_x
food_distance_y
snake_length_normalized
steps_since_food_normalized

CNN-DQN state

The CNN-DQN agent uses a grid state:

shape = 3 x rows x cols

Channels:

channel 0 = snake head
channel 1 = snake body
channel 2 = food

Install dependencies

From the project root:

pip install -r snake_ai/requirements.txt

Play manually

python snake_ai/src/play.py --mode human

Controls:

Arrow keys or WASD = move
ESC = quit

Run random agent

python snake_ai/src/play.py --mode random

Train MLP-DQN

python snake_ai/src/train_mlp_dqn.py --episodes 500 --render-every 50

Without rendering:

python snake_ai/src/train_mlp_dqn.py --episodes 500 --render-every 0

Resume training:

python snake_ai/src/train_mlp_dqn.py --resume snake_ai/checkpoints/mlp_dqn_latest.pth

Train CNN-DQN

python snake_ai/src/train_cnn_dqn.py --episodes 500 --render-every 50

Without rendering:

python snake_ai/src/train_cnn_dqn.py --episodes 500 --render-every 0

Resume training:

python snake_ai/src/train_cnn_dqn.py --resume snake_ai/checkpoints/cnn_dqn_latest.pth

Run tests

pytest snake_ai/tests

Development plan

Stage 1:

* Build pygame Snake environment
* Support human play
* Support random agent
* Add basic tests

Stage 2:

* Train MLP-DQN with vector state
* Save and resume checkpoints
* Track score, loss, epsilon, and training time

Stage 3:

* Train CNN-DQN with grid state
* Compare learning speed and final performance with MLP-DQN

Stage 4:

* Add training plots
* Add evaluation mode
* Improve reward design
* Try Double DQN or Dueling DQN

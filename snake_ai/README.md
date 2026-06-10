# Snake AI

This project explores how to train AI agents to play Snake.

The goal is to build a reusable Snake game environment with `pygame`, then test different algorithms and reward functions on the same environment.

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
│   ├── rewards/
│   │   └── reward_functions.py
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

---

## 1. Install dependencies

From the project root:

```bash
pip install -r snake_ai/requirements.txt
```

If you are using a conda environment:

```bash
conda activate noobs
pip install -r snake_ai/requirements.txt
```

For Windows users with NVIDIA GPU, install CUDA-enabled PyTorch separately:

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

Check whether CUDA is available:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

Expected CUDA result:

```text
True
NVIDIA GeForce ...
```

On Apple Silicon Mac, the program may use MPS automatically:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

---

## 2. Play Snake manually

Run:

```bash
python snake_ai/src/play.py --mode human
```

Controls:

```text
Arrow keys or WASD = move
ESC = quit
```

---

## 3. Run the random agent

This is mainly used to check whether the environment works.

```bash
python snake_ai/src/play.py --mode random
```

---

## 4. Train MLP-DQN

Basic training command:

```bash
python snake_ai/src/train_mlp_dqn.py --episodes 500 --render-every 0 --save-every 50
```

This trains the MLP-DQN agent using the default reward mode:

```text
classic
```

The default checkpoint is saved to:

```text
snake_ai/checkpoints/mlp_dqn_latest.pth
```

---

## 5. MLP-DQN training parameters

The training script supports these parameters:

```text
--episodes        Number of training episodes
--render-every    Render one episode every N episodes
--save-every      Save checkpoint every N episodes
--reward          Choose reward function
--checkpoint      Path to save checkpoint
--resume          Resume training from an existing checkpoint
```

Example:

```bash
python snake_ai/src/train_mlp_dqn.py \
  --episodes 500 \
  --render-every 0 \
  --save-every 50 \
  --reward classic \
  --checkpoint snake_ai/checkpoints/mlp_classic_latest.pth
```

Parameter explanation:

```text
--episodes 500
Train for 500 games.

--render-every 0
Do not show pygame window during training. This is much faster.

--render-every 50
Show one visualized game every 50 episodes.

--save-every 50
Save model checkpoint every 50 episodes.

--reward classic
Use the classic reward function.

--checkpoint snake_ai/checkpoints/mlp_classic_latest.pth
Save the trained model to this path.

--resume snake_ai/checkpoints/mlp_classic_latest.pth
Continue training from an existing model.
```

---

## 6. Available reward functions

Reward functions are defined in:

```text
snake_ai/src/rewards/reward_functions.py
```

Current reward modes:

```text
classic
distance
step_penalty
anti_loop
```

### classic

Original sparse reward:

```text
Eat food: +10
Die: -10
Normal move: 0
```

Command:

```bash
python snake_ai/src/train_mlp_dqn.py \
  --episodes 500 \
  --render-every 0 \
  --save-every 50 \
  --reward classic \
  --checkpoint snake_ai/checkpoints/mlp_classic_latest.pth
```

### distance

Reward based on whether the snake moves closer to food:

```text
Eat food: +10
Die: -10
Move closer to food: +0.1
Move farther from food: -0.1
Same distance: 0
```

Command:

```bash
python snake_ai/src/train_mlp_dqn.py \
  --episodes 500 \
  --render-every 0 \
  --save-every 50 \
  --reward distance \
  --checkpoint snake_ai/checkpoints/mlp_distance_latest.pth
```

### step_penalty

Adds a small penalty for every step to encourage faster food collection:

```text
Eat food: +10
Die: -10
Every normal step: -0.01
Move closer to food: additional +0.1
Move farther from food: additional -0.1
```

Actual normal movement rewards:

```text
Closer to food: +0.09
Farther from food: -0.11
Same distance: -0.01
```

Command:

```bash
python snake_ai/src/train_mlp_dqn.py \
  --episodes 500 \
  --render-every 0 \
  --save-every 50 \
  --reward step_penalty \
  --checkpoint snake_ai/checkpoints/mlp_step_penalty_latest.pth
```

### anti_loop

Designed to reduce useless circling behavior.

It is based on `step_penalty`, but adds extra punishment when the snake has not eaten food for a long time:

```text
Eat food: +10
Die: -10
Every normal step: -0.01
Move closer to food: additional +0.1
Move farther from food: additional -0.1
After 50% of max_steps_without_food: additional -0.02
After 80% of max_steps_without_food: additional -0.05
```

Command:

```bash
python snake_ai/src/train_mlp_dqn.py \
  --episodes 500 \
  --render-every 0 \
  --save-every 50 \
  --reward anti_loop \
  --checkpoint snake_ai/checkpoints/mlp_anti_loop_latest.pth
```

---

## 7. Compare four reward functions

Run these four commands separately.

### Train classic

```bash
python snake_ai/src/train_mlp_dqn.py \
  --episodes 500 \
  --render-every 0 \
  --save-every 50 \
  --reward classic \
  --checkpoint snake_ai/checkpoints/mlp_classic_latest.pth
```

### Train distance

```bash
python snake_ai/src/train_mlp_dqn.py \
  --episodes 500 \
  --render-every 0 \
  --save-every 50 \
  --reward distance \
  --checkpoint snake_ai/checkpoints/mlp_distance_latest.pth
```

### Train step_penalty

```bash
python snake_ai/src/train_mlp_dqn.py \
  --episodes 500 \
  --render-every 0 \
  --save-every 50 \
  --reward step_penalty \
  --checkpoint snake_ai/checkpoints/mlp_step_penalty_latest.pth
```

### Train anti_loop

```bash
python snake_ai/src/train_mlp_dqn.py \
  --episodes 500 \
  --render-every 0 \
  --save-every 50 \
  --reward anti_loop \
  --checkpoint snake_ai/checkpoints/mlp_anti_loop_latest.pth
```

During comparison, focus on:

```text
Record
Avg100
Whether the snake circles uselessly
How fast it reaches food
How often it dies by hitting wall or itself
```

---

## 8. Continue training from a checkpoint

Example:

```bash
python snake_ai/src/train_mlp_dqn.py \
  --episodes 500 \
  --render-every 0 \
  --save-every 50 \
  --reward classic \
  --checkpoint snake_ai/checkpoints/mlp_classic_latest.pth \
  --resume snake_ai/checkpoints/mlp_classic_latest.pth
```

Important:

```text
The reward mode used during resume should match the checkpoint.
```

For example, do not resume an `anti_loop` experiment from a `classic` checkpoint unless you intentionally want to fine-tune from another reward setup.

---

## 9. Watch trained models play

Use `play.py` with `--mode mlp` and specify the checkpoint.

### Watch classic model

```bash
python snake_ai/src/play.py \
  --mode mlp \
  --checkpoint snake_ai/checkpoints/mlp_classic_latest.pth
```

### Watch distance model

```bash
python snake_ai/src/play.py \
  --mode mlp \
  --checkpoint snake_ai/checkpoints/mlp_distance_latest.pth
```

### Watch step_penalty model

```bash
python snake_ai/src/play.py \
  --mode mlp \
  --checkpoint snake_ai/checkpoints/mlp_step_penalty_latest.pth
```

### Watch anti_loop model

```bash
python snake_ai/src/play.py \
  --mode mlp \
  --checkpoint snake_ai/checkpoints/mlp_anti_loop_latest.pth
```

---

## 10. Train CNN-DQN

CNN-DQN uses grid state instead of the 19-dimensional vector state.

Basic command:

```bash
python snake_ai/src/train_cnn_dqn.py --episodes 500 --render-every 0 --save-every 50
```

Example with a separate checkpoint:

```bash
python snake_ai/src/train_cnn_dqn.py \
  --episodes 500 \
  --render-every 0 \
  --save-every 50 \
  --checkpoint snake_ai/checkpoints/cnn_dqn_latest.pth
```

Watch CNN-DQN model:

```bash
python snake_ai/src/play.py \
  --mode cnn \
  --checkpoint snake_ai/checkpoints/cnn_dqn_latest.pth
```

Note:

```text
CNN-DQN is usually slower to train than MLP-DQN.
Start with MLP-DQN first when testing reward functions.
```

---

## 11. Where to change reward functions

Reward functions are stored in:

```text
snake_ai/src/rewards/reward_functions.py
```

To add a new reward mode:

1. Define a new function.
2. Add it to `REWARD_FUNCTIONS`.

Example:

```python
def reward_my_new_mode(
    ate_food: bool,
    died: bool,
    old_distance_to_food: int,
    new_distance_to_food: int,
    steps_since_food: int,
    max_steps_without_food: int,
) -> float:
    if died:
        return -10.0

    if ate_food:
        return 10.0

    return 0.0


REWARD_FUNCTIONS = {
    "classic": reward_classic,
    "distance": reward_distance,
    "step_penalty": reward_step_penalty,
    "anti_loop": reward_anti_loop,
    "my_new_mode": reward_my_new_mode,
}
```

Then train with:

```bash
python snake_ai/src/train_mlp_dqn.py \
  --episodes 500 \
  --render-every 0 \
  --reward my_new_mode \
  --checkpoint snake_ai/checkpoints/mlp_my_new_mode_latest.pth
```

---

## 12. Run tests

Run all tests:

```bash
pytest snake_ai/tests -v
```

Run specific tests:

```bash
pytest snake_ai/tests/test_snake_env.py -v
pytest snake_ai/tests/test_state_vector.py -v
pytest snake_ai/tests/test_replay_buffer.py -v
pytest snake_ai/tests/test_models.py -v
```

---

## 13. Git note

Model checkpoints can be large and change frequently. Usually, do not commit `.pth` files.

Recommended `.gitignore` entries:

```gitignore
snake_ai/checkpoints/*.pth
snake_ai/runs/
__pycache__/
*.pyc
```

Commit source code only:

```bash
git add snake_ai .gitignore
git commit -m "update snake ai usage guide and reward experiments"
git push origin main
```
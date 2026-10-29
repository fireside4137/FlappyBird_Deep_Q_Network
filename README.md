# Flappy Bird Deep Q-Learning (DQN & Dueling DQN)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-FlappyBird--v0-brightgreen.svg)](https://github.com/Farama-Foundation/Gymnasium)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Deep Reinforcement Learning project training an autonomous agent to master **Flappy Bird** using **Deep Q-Networks (DQN)** and **Dueling DQN** in PyTorch.

---

## 🌟 Highlights

- **Dual Architectures**: Choose between **Standard DQN** and **Dueling DQN** (decoupling State-Value $V(s)$ and Advantage $A(s, a)$).
- **High Sample Efficiency**: Per-step mini-batch updates, replay buffer warm-up phase, Huber Loss (`SmoothL1Loss`), and gradient clipping.
- **Hardware-Accelerated**: Automatic device detection across Apple Silicon (`mps`), NVIDIA GPUs (`cuda`), and CPU.
- **Memory Optimized**: Replay buffer stores lightweight NumPy primitives, avoiding GPU memory leaks.
- **Evaluation & Video Capture**: Dedicated evaluation pipeline with real-time Pygame rendering or headless MP4 recording (`gym.wrappers.RecordVideo`).
- **Training Analytics**: Automated CSV metric logging (reward, rolling average, epsilon, TD-loss) and publication-ready curve plotting.
- **Pre-trained Model Included**: Test a trained agent right after cloning!

---

## 🧠 Architectures

### 1. Standard DQN
Maps the 12-dimensional state vector directly to action Q-values through a feed-forward neural network:
```
State (12) ──► Linear(12, 256) ──► ReLU ──► Linear(256, 2) ──► Q(s, a) [0: Idle, 1: Flap]
```

### 2. Dueling DQN
Separates the state representation into two independent streams:
1. **Value Stream $V(s)$**: Assesses how good the state is overall (scalar).
2. **Advantage Stream $A(s, a)$**: Evaluates the relative advantage of flapping vs. idling in that state.

```
                                      ┌──► Value Stream V(s) (Scalar) ────────┐
State (12) ──► Shared Linear(12, 256)                                         ├──► Q(s, a)
                                      └──► Advantage Stream A(s, a) (Vector) ─┘
```

The streams are aggregated using mean-subtracted advantage for numerical stability:
$$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)$$

---

## 📁 Project Structure

```
Flappybird_DeepQN/
├── parameters.yaml        # Hyperparameter profiles (Standard DQN & Dueling DQN)
├── dqn.py                 # DQN and DuelingDQN neural network architectures
├── experience_replay.py   # FIFO Replay Memory buffer
├── agent.py               # Core DQN training and inference engine
├── evaluate.py            # Evaluation benchmark & MP4 video recorder
├── plot_results.py        # Training curve visualization tool
├── requirements.txt       # Project dependencies
├── LICENSE                # MIT License
├── .gitignore             # Git exclusions
└── runs/                  # Artifacts directory
    ├── flappybirdv0.pt    # Pre-trained model checkpoint
    └── flappybirdv0.log   # Training log history
```

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fireside4137/Flappybird_DeepQN.git
   cd Flappybird_DeepQN
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   conda create -n flappy_dqn python=3.10 -y
   conda activate flappy_dqn
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Quick Start & Usage

### 1. Test Pre-trained Model Immediately
Watch the pre-trained agent play Flappy Bird with live rendering:
```bash
python evaluate.py flappybirdv0 --episodes 5
```

### 2. Train an Agent

- **Train Dueling DQN (Recommended):**
  ```bash
  python agent.py flappybird_dueling --train
  ```

- **Train Standard DQN:**
  ```bash
  python agent.py flappybirdv0 --train
  ```

- **Train for a specific number of episodes:**
  ```bash
  python agent.py flappybird_dueling --train --episodes 1000
  ```

### 3. Record Gameplay Videos
Record high-definition MP4 videos of evaluation episodes saved to `runs/videos/`:
```bash
python evaluate.py flappybirdv0 --episodes 3 --record
```

### 4. Plot Training Curves
Visualize training progress (Episode rewards, 100-episode moving average, epsilon exploration decay, and training loss):
```bash
python plot_results.py flappybird_dueling
```
*The resulting plot will be saved to `runs/<param_set>_training_curve.png`.*

---

## ⚙️ Hyperparameter Configuration (`parameters.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `network_type` | `dueling` / `standard` | Neural network architecture |
| `alpha` | `0.0005` | Learning rate for the Adam optimizer |
| `gamma` | `0.99` | Discount factor for future rewards |
| `epsilon_init` | `1.0` | Initial exploration rate |
| `epsilon_min` | `0.05` | Minimum exploration floor |
| `epsilon_decay` | `0.995` | Multiplicative decay per episode after warm-up |
| `replay_memory_size` | `100,000` | Maximum capacity of the experience buffer |
| `min_replay_size` | `1,000` | Warm-up transitions collected before training begins |
| `mini_batch_size` | `32` | Batch size sampled for gradient updates |
| `network_sync_rate` | `1,000` | Steps between updating target network weights |
| `reward_threshold` | `1,000` | Score threshold to terminate training |

---

## 📊 State Representation (12 Dimensions)

The `FlappyBird-v0` environment provides a normalized 12-dimensional continuous state vector:
1. Horizontal distance to the last pipe
2. Vertical distance to the top pipe of the last pair
3. Vertical distance to the bottom pipe of the last pair
4. Horizontal distance to the next pipe
5. Vertical distance to the top pipe of the next pair
6. Vertical distance to the bottom pipe of the next pair
7. Horizontal distance to the next-next pipe
8. Vertical distance to the top pipe of the next-next pair
9. Vertical distance to the bottom pipe of the next-next pair
10. Bird vertical position $y$
11. Bird vertical velocity $v_y$
12. Bird rotation angle $\theta$

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

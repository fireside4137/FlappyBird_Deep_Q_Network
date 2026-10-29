import flappy_bird_gymnasium
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import yaml
import random
import argparse
import os
from collections import deque
import numpy as np

from dqn import DQN
from experience_replay import ReplayMemory


# Selecting the device for the model
if torch.backends.mps.is_available():
    device = "mps"          # apple silicon
elif torch.cuda.is_available():
    device = "cuda"         # Nvidia GPU
else:
    device = "cpu"          # normal CPU

RUNS_DIR = "runs" # folder to store the best params of a run
os.makedirs(RUNS_DIR, exist_ok = True)


class Agent:

    def __init__(self, param_set):

        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.env_id = params.get("env_id", "FlappyBird-v0")
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.min_replay_size = params.get("min_replay_size", 1000)
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]

        self.reward_threshold = params["reward_threshold"]

        # Smooth L1 (Huber) loss is much more robust to outliers and TD-error spikes than MSE
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = None

        self.LOG_FILES = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")


    def run(self, is_training = True, render = False):

        env = gym.make(self.env_id, render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]    # input dim
        num_actions = env.action_space.n               # output dim

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            target_dqn.eval()

            total_steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr = self.alpha)
            best_reward = float("-inf")
            reward_history = deque(maxlen=100)

        else:
            # Load trained policy for evaluation
            if not os.path.exists(self.MODEL_FILE):
                raise FileNotFoundError(f"Model file '{self.MODEL_FILE}' not found. Please train the model first.")
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()

        try:
            for episode in itertools.count():

                state, _ = env.reset()
                episode_reward = 0
                terminated = False
                truncated = False

                while not (terminated or truncated) and episode_reward < self.reward_threshold:
                    # Action selection (epsilon-greedy)
                    if is_training and random.random() < epsilon:
                        action = env.action_space.sample() # explore
                    else:
                        with torch.no_grad():
                            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                            action = policy_dqn(state_tensor).argmax().item() # exploit

                    # Step environment
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward

                    if is_training:
                        # Store raw numpy / python types to prevent GPU memory leaks
                        memory.append((state, action, next_state, reward, terminated))
                        total_steps += 1

                        # Train every step once replay buffer has sufficient warm-up transitions
                        if len(memory) >= self.min_replay_size:
                            mini_batch = memory.sample(self.mini_batch_size)
                            self.optimize(mini_batch, policy_dqn, target_dqn)

                            # Periodically sync target network with policy network
                            if total_steps % self.network_sync_rate == 0:
                                target_dqn.load_state_dict(policy_dqn.state_dict())

                    state = next_state

                if is_training:
                    # Decay epsilon once training has started
                    if len(memory) >= self.min_replay_size:
                        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                    reward_history.append(episode_reward)
                    avg_reward_100 = sum(reward_history) / len(reward_history)

                    if episode_reward > best_reward:
                        log_msg = f"best reward = {episode_reward} for episode = {episode+1}"
                        with open(self.LOG_FILES, "a") as f:
                            f.write(log_msg + "\n")

                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        best_reward = episode_reward

                    # Log progress
                    status = "TRAINING" if len(memory) >= self.min_replay_size else f"WARMUP ({len(memory)}/{self.min_replay_size})"
                    print(f"episode={episode+1}, reward={episode_reward:.2f}, avg_100={avg_reward_100:.2f}, epsilon={epsilon:.4f}, [{status}]")

                else:
                    print(f"episode={episode+1}, reward={episode_reward:.2f}")

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            env.close()


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Unpack experiences
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        # Batch convert directly to device tensors efficiently
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        terminations = torch.tensor(terminations, dtype=torch.float32, device=device)

        # Calculate target Q-values: y = r + (1 - done) * gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        # Calculate current predicted Q-values: Q_policy(s, a)
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze(-1)

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()


if __name__ == "__main__":
    # parse command line inputs
    parser = argparse.ArgumentParser(description = 'Train or test model.')
    parser.add_argument('hyperparameters', help = '')
    parser.add_argument('--train', help = 'Training mode', action = 'store_true')
    args = parser.parse_args()

    dql = Agent(param_set = args.hyperparameters)

    if args.train:
        dql.run(is_training = True)
    else:
        dql.run(is_training = False, render = True)

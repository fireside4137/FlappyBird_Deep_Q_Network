import flappy_bird_gymnasium
import gymnasium as gym
import torch
import itertools
import yaml
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import os
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

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]

        self.reward_threshold = params["reward_threshold"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILES = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")




    def run(self, is_training = True, render = False):

        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]    #input dim
        num_actions = env.action_space.n               #output dim

        policy_dqn = DQN(num_states, num_actions).to(device)


        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            # copy the wt and bias vals from policy to target
            target_dqn.load_state_dict(policy_dqn.state_dict())
            target_dqn.eval()

            steps = 0

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr = self.alpha)

            best_reward = float("-inf")

        else :
            # best policy load
            if not os.path.exists(self.MODEL_FILE):
                raise FileNotFoundError(f"Model file '{self.MODEL_FILE}' not found. Please train the model first.")
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()

        try:
            for episode in itertools.count():

                state, _ = env.reset()
                state = torch.tensor(state, dtype = torch.float, device = device)

                episode_reward = 0
                terminated = False
                truncated = False

                while not (terminated or truncated) and episode_reward < self.reward_threshold:
                    # Next action:
                    if is_training and random.random() < epsilon:
                        action = env.action_space.sample() # explore
                        action = torch.tensor(action, dtype = torch.long, device = device)
                    else:
                        with torch.no_grad():
                            action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() # exploit


                    # Processing: terminated = done
                    next_state, reward, terminated, truncated, _ = env.step(action.item())
                    episode_reward += reward

                    # create tensors
                    reward = torch.tensor(reward, dtype=torch.float, device = device)
                    next_state = torch.tensor(next_state, dtype=torch.float, device = device)

                    if is_training:
                        memory.append((state, action, next_state, reward, terminated))
                        steps += 1

                    state = next_state

                if is_training:
                    # epsilon decay 
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                    if episode_reward > best_reward:
                        log_msg = f"best reward = {episode_reward} for episode = {episode+1}"

                        with open(self.LOG_FILES, "a") as f:
                            f.write(log_msg + "\n")

                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        best_reward = episode_reward

                    print(f"episode={episode+1}, reward={episode_reward:.2f}, epsilon={epsilon:.4f}")

                    if len(memory) > self.mini_batch_size:
                        # get samples
                        mini_batch = memory.sample(self.mini_batch_size)

                        self.optimize(mini_batch, policy_dqn, target_dqn)

                        # sync the network
                        if steps > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            steps = 0
                else:
                    print(f"episode={episode+1}, reward={episode_reward:.2f}")

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            env.close()


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # get experience => batch training to make this training faster
        # for state, action, next_state, reward, terminated in mini_batch:

        #     if terminated : 
        #         target = reward

        #     else:
        #         with torch.no_grad():
        #             target_q = reward + self.gamma * target_dqn(next_state).max()

        #     current_q = policy_dqn(state)

        #     # loss
        #     loss = self.loss_fn(current_q, target_q)

        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()

        # => this is the optimized version of above code, in which the experience is shared in batches 
        # making the training faster

        # get the batch of experiences
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        # calculate target Q-values - if terminations = true => zero
        # if terminations = true, => 1-1 = 0, and everything becomes zero, else everyting runs.
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        # calculate y_pred i.e. Q-value from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze(-1)

        # compute loss
        loss = self.loss_fn(current_q, target_q)

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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
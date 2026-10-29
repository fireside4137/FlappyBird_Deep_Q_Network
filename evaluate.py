import argparse
import os
import yaml
import torch
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium

from dqn import DQN, DuelingDQN


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

RUNS_DIR = "runs"
VIDEO_DIR = os.path.join(RUNS_DIR, "videos")


def load_model(param_set, model_path=None):
    """Load model architecture and weights based on parameters.yaml."""
    with open("parameters.yaml", "r") as f:
        params = yaml.safe_load(f)[param_set]

    env_id = params.get("env_id", "FlappyBird-v0")
    network_type = params.get("network_type", "standard").lower()

    if model_path is None:
        model_path = os.path.join(RUNS_DIR, f"{param_set}.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at '{model_path}'. Please train the model first.")

    # Determine input and output dimensions
    dummy_env = gym.make(env_id)
    num_states = dummy_env.observation_space.shape[0]
    num_actions = dummy_env.action_space.n
    dummy_env.close()

    # Instantiate model
    if network_type == "dueling":
        model = DuelingDQN(num_states, num_actions).to(device)
    else:
        model = DQN(num_states, num_actions).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, env_id, network_type


def evaluate(param_set, episodes=10, record=False, render=True, model_path=None):
    """Evaluate a trained model over multiple episodes and compute statistics."""
    model, env_id, network_type = load_model(param_set, model_path)

    print(f"\n{'=' * 60}")
    print(f" EVALUATION: {param_set}")
    print(f" Architecture: {network_type.upper()} DQN | Device: {device}")
    print(f" Episodes: {episodes} | Recording: {record}")
    print(f"{'=' * 60}\n")

    # Render mode selection
    if record:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        env = gym.make(env_id, render_mode="rgb_array")
        # Record all evaluation episodes
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=VIDEO_DIR,
            name_prefix=f"{param_set}_eval",
            episode_trigger=lambda ep_idx: True
        )
    else:
        env = gym.make(env_id, render_mode="human" if render else None)

    episode_rewards = []
    episode_steps = []

    try:
        for ep in range(episodes):
            state, _ = env.reset()
            ep_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = model(state_tensor).argmax().item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                steps += 1
                state = next_state

            episode_rewards.append(ep_reward)
            episode_steps.append(steps)
            print(f" Episode {ep + 1:2d}/{episodes}: Reward = {ep_reward:7.2f} | Steps = {steps:4d}")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        env.close()

    if episode_rewards:
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)
        mean_steps = np.mean(episode_steps)

        print(f"\n{'-' * 60}")
        print(" EVALUATION SUMMARY")
        print(f"{'-' * 60}")
        print(f" Completed Episodes:  {len(episode_rewards)}/{episodes}")
        print(f" Mean Reward:         {mean_reward:7.2f} +/- {std_reward:5.2f}")
        print(f" Max Reward:          {max_reward:7.2f}")
        print(f" Min Reward:          {min_reward:7.2f}")
        print(f" Average Steps/Ep:    {mean_steps:7.1f}")
        if record:
            print(f" Videos saved to:     {VIDEO_DIR}/")
        print(f"{'=' * 60}\n")

    return episode_rewards, episode_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN or Dueling DQN agent on Flappy Bird.")
    parser.add_argument("hyperparameters", help="Hyperparameter profile from parameters.yaml (e.g. flappybirdv0)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes (default: 10)")
    parser.add_argument("--record", action="store_true", help="Record video of playthroughs to runs/videos/")
    parser.add_argument("--no-render", action="store_true", help="Disable GUI rendering for faster headless testing")
    parser.add_argument("--model-path", type=str, default=None, help="Custom path to model weights .pt file")

    args = parser.parse_args()

    evaluate(
        param_set=args.hyperparameters,
        episodes=args.episodes,
        record=args.record,
        render=not args.no_render,
        model_path=args.model_path
    )

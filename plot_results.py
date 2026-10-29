import argparse
import os
import re
import matplotlib
# Use Agg backend by default so plotting works on headless environments or terminal
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = "runs"


def parse_csv_stats(stats_path):
    """Parse stats CSV file generated during training."""
    episodes = []
    rewards = []
    avg_rewards = []
    epsilons = []
    losses = []

    with open(stats_path, "r") as f:
        lines = f.readlines()

    header = [h.strip() for h in lines[0].split(",")]

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        episodes.append(int(parts[0]))
        rewards.append(float(parts[1]))
        avg_rewards.append(float(parts[2]))
        epsilons.append(float(parts[3]))
        losses.append(float(parts[4]))

    return {
        "episodes": np.array(episodes),
        "rewards": np.array(rewards),
        "avg_rewards": np.array(avg_rewards),
        "epsilons": np.array(epsilons),
        "losses": np.array(losses),
    }


def parse_best_log(log_path):
    """Fallback parser for legacy best reward log files."""
    episodes = []
    best_rewards = []

    pattern = re.compile(r"best reward\s*=\s*([\d\.\-]+)\s*for episode\s*=\s*(\d+)")

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                best_rewards.append(float(match.group(1)))
                episodes.append(int(match.group(2)))

    return {
        "episodes": np.array(episodes),
        "best_rewards": np.array(best_rewards)
    }


def plot_training_results(param_set, save_path=None, show=False):
    """Generate training visualization plots."""
    stats_file = os.path.join(RUNS_DIR, f"{param_set}_stats.csv")
    log_file = os.path.join(RUNS_DIR, f"{param_set}.log")

    if save_path is None:
        save_path = os.path.join(RUNS_DIR, f"{param_set}_training_curve.png")

    if os.path.exists(stats_file):
        data = parse_csv_stats(stats_file)
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle(f"Training Progress: {param_set}", fontsize=14, fontweight="bold")

        # Subplot 1: Episode Reward & Moving Average
        ax1 = axes[0]
        ax1.plot(data["episodes"], data["rewards"], label="Episode Reward", color="#4C72B0", alpha=0.35, linewidth=0.8)
        ax1.plot(data["episodes"], data["avg_rewards"], label="100-Ep Moving Avg", color="#C44E52", linewidth=2.0)
        ax1.set_ylabel("Reward")
        ax1.set_title("Reward per Episode")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper left")

        # Subplot 2: Epsilon Decay
        ax2 = axes[1]
        ax2.plot(data["episodes"], data["epsilons"], label="Epsilon (Exploration)", color="#55A868", linewidth=1.5)
        ax2.set_ylabel("Epsilon")
        ax2.set_title("Exploration Rate Decay")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper right")

        # Subplot 3: Loss Curve
        ax3 = axes[2]
        ax3.plot(data["episodes"], data["losses"], label="TD Loss (Huber)", color="#8172B2", linewidth=1.2)
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Loss")
        ax3.set_title("Average Training Loss per Episode")
        ax3.grid(True, linestyle="--", alpha=0.6)
        ax3.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f" Training curve saved to: {save_path}")

        print(f"\nTraining Stats:")
        print(f"  Total Episodes: {data['episodes'][-1] if len(data['episodes']) else 0}")
        print(f"  Max Reward:     {data['rewards'].max():.2f}")
        print(f"  Final Avg 100:  {data['avg_rewards'][-1]:.2f}")
        print(f"  Final Epsilon:  {data['epsilons'][-1]:.4f}")

    elif os.path.exists(log_file):
        data = parse_best_log(log_file)
        if len(data["episodes"]) == 0:
            print(f"No valid log entries found in '{log_file}'.")
            return

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(data["episodes"], data["best_rewards"], marker="o", markersize=4, color="#C44E52", linewidth=1.8, label="Best Reward Trajectory")
        ax.set_title(f"Best Reward Progression: {param_set}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Best Reward Achieved")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f" Training curve plotted from best log and saved to: {save_path}")
        print(f"\nStats from Log:")
        print(f"  Best Reward Logged: {data['best_rewards'][-1]:.2f} at episode {data['episodes'][-1]}")

    else:
        print(f"No stats CSV ({stats_file}) or log ({log_file}) found for '{param_set}'.")
        return

    if show:
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curves for Flappy Bird DQN / Dueling DQN.")
    parser.add_argument("hyperparameters", help="Hyperparameter profile name (e.g. flappybirdv0)")
    parser.add_argument("--save-path", type=str, default=None, help="Custom path to save the output PNG")
    parser.add_argument("--show", action="store_true", help="Display the plot in an interactive window")

    args = parser.parse_args()
    plot_training_results(param_set=args.hyperparameters, save_path=args.save_path, show=args.show)

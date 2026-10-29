import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Standard Deep Q-Network with a single feed-forward trunk.
    """
    def __init__(self, state_dim=12, action_dim=2, hidden_dim=256):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.model(x)


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network:
    Decouples state-value estimation V(s) and action-advantage estimation A(s, a).
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
    """
    def __init__(self, state_dim=12, action_dim=2, hidden_dim=256):
        super(DuelingDQN, self).__init__()

        # Shared feature representation
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # State-value stream V(s) -> scalar
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Action-advantage stream A(s, a) -> vector of size action_dim
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values

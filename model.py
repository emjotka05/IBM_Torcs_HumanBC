"""
Policy network for the imitation driver.

A small feed-forward net maps a normalised observation to three continuous
controls. Steering is squashed to [-1, 1]; throttle and brake to [0, 1].
The shared trunk keeps the heads cheap and lets one backbone serve all three.
"""

import torch
import torch.nn as nn


class ClonePolicy(nn.Module):
    """Observation -> (steer, throttle, brake)."""

    def __init__(self, n_features=17):
        super().__init__()
        # Shared trunk: 3 ReLU layers, narrowing 256 -> 128 -> 64.
        self.fc = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # One linear head per control channel.
        self.steer_head = nn.Linear(64, 1)
        self.accel_head = nn.Linear(64, 1)
        self.brake_head = nn.Linear(64, 1)

    def forward(self, obs):
        trunk = self.fc(obs)
        steer = torch.tanh(self.steer_head(trunk))      # [-1, 1]
        throttle = torch.sigmoid(self.accel_head(trunk))  # [0, 1]
        brake = torch.sigmoid(self.brake_head(trunk))   # [0, 1]
        return torch.cat([steer, throttle, brake], dim=1)


class ValueHead(nn.Module):
    """Optional state-action value estimator (not used by behavioral cloning)."""

    def __init__(self, n_features=17, n_controls=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_features + n_controls, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs, controls):
        return self.fc(torch.cat([obs, controls], dim=1))

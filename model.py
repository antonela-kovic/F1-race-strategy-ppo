# model.py
# NOTE: LSTMFeatureExtractor is currently NOT used. train.py uses a standard
# MLP architecture (net_arch=[256, 128, 64]) defined directly in the PPO call.
# This class is kept as a starting point for future work on a recurrent policy.
# It does not implement real memory across laps: every forward() call builds a
# new sequence of length 1 with a zero hidden state. Real memory over time
# would require RecurrentPPO (sb3-contrib) instead of the standard PPO.
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=features_dim, batch_first=True)
        self._features_dim = features_dim

    def forward(self, obs):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        _, (h_n, _) = self.lstm(obs)
        return h_n.squeeze(0)

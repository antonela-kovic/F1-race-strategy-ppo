# === model.py ===
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class RNDModel(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim)
        )
        for p in self.target.parameters():
            p.requires_grad = False
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()

    def compute_intrinsic_reward(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            target_out = self.target(obs_tensor)
        pred_out = self.predictor(obs_tensor)
        return torch.mean((pred_out - target_out) ** 2, dim=1).detach().numpy()

    def train_predictor(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        target_out = self.target(obs_tensor)
        pred_out = self.predictor(obs_tensor)
        loss = self.loss_fn(pred_out, target_out)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

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


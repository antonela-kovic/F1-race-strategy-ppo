# === model.py ===
# NAPOMENA: klasa ispod (LSTMFeatureExtractor) trenutno se NE koristi u
# train.py - taj koristi standardnu MLP arhitekturu (net_arch=[256,128,64])
# definiranu izravno u PPO pozivu, u skladu s onim što opisuje pisani rad.
# Ova klasa je ovdje sačuvana kao polazna točka za budući rad prema
# rekurentnoj politici. VAŽNO ako je se ikad ponovno poveže: ispod NIJE
# implementirana stvarna memorija kroz krugove - svaki forward() poziv gradi
# novu "sekvencu" duljine 1 s nultim skrivenim stanjem, pa LSTM sloj ovdje
# efektivno ne pamti prethodne krugove. Za stvarnu memoriju kroz vrijeme
# potreban je RecurrentPPO (paket sb3-contrib), ne standardni PPO.
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



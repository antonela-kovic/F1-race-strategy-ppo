# test_env.py --> okolina za brzo testiranje
import numpy as np
from stable_baselines3 import PPO
from f1_env import FastF1MultiActionEnv
from utils import load_obs_array, load_true_actions

# Učitavanje podataka
obs_array = load_obs_array("data/obs_array.npy")
true_actions = load_true_actions("data/true_actions.npy")

# Inicijalizacija okruženja
env = FastF1MultiActionEnv(obs_array, true_actions)

# Učitavanje treniranog modela
model = PPO.load("saved_models/ppo_f1_rl.zip")

# Testiranje modela
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    print(f"Action: {action}, Reward: {reward:.2f}")

print(f"\nTotal reward: {total_reward:.2f}")

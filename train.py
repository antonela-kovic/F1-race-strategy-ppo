# === train.py ===
from stable_baselines3 import PPO
from f1_env import FastF1MultiActionEnv
from model import LSTMFeatureExtractor
import matplotlib.pyplot as plt
from utils import selected_features
import numpy as np
from collections import Counter

obs_array = np.load("data/obs_array.npy")
true_actions = np.load("data/true_actions.npy")

env = FastF1MultiActionEnv(obs_array, true_actions=true_actions, difficulty_level="easy")

# Nadogradnja: ent_coef je PPO hyperparametar (entropy bonus) i ide kao argument u konstruktor PPO(...), ne u learn().
# model = PPO("MlpPolicy", env, policy_kwargs=dict(
#     features_extractor_class=LSTMFeatureExtractor,
#     features_extractor_kwargs=dict(features_dim=64)), verbose=1)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64)
    ),
    ent_coef=0.02,     # <-- OVDJE
    verbose=1
)

rewards = []
weather_rewards = {"suho": [], "mokro": [], "kiša": []}

for i in range(100):
    # === Curriculum s metodom ===
    if i in [30, 60]:
        env.increase_difficulty()
        print(f"Težina promijenjena u: {env.difficulty_level}")

    model.learn(total_timesteps=2000, reset_num_timesteps=False)
    obs, _ = env.reset(seed=42)

    # === Detaljna vremenska distribucija ===
    weather_map = {1: "suho", 2: "mokro", 3: "kiša"}
    weather_human = [weather_map.get(row[selected_features.index("Weather")], "?") for row in env.filtered_obs_array]
    print(f"Epizoda {i+1} – Vrijeme po krugovima:", weather_human[:10], "...")

    transitions = sum(weather_human[j] != weather_human[j-1] for j in range(1, len(weather_human)))
    print(f"Promjena vremenskih uvjeta: {transitions}x")

    total_reward = 0
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        if steps < len(weather_human):
            current_weather = weather_human[steps]
            if current_weather in weather_rewards:
                weather_rewards[current_weather].append(reward)

        steps += 1
        if done:
            print(f"Epizoda {i+1} završila nakon {steps} koraka (reward = {total_reward:.2f})")
            break

    # === Smoothed reward ===
    smoothed = 0.95 * rewards[-1] + 0.05 * total_reward if rewards else total_reward
    rewards.append(smoothed)



#model.save("saved_models/ppo_f1_rl.zip")
#model.save("saved_models/ppo_f1_rl_logic_v2.zip") # novi naziv za model kako nebi pregazili stari model koji nije imao ent_coef i bez logic_layer 
model.save("saved_models/ppo_f1_rl_logic_v4_retrained.zip") # verzija kod koje popravljamo pit stop u f1_env.py
# === Vizualizacija 1: Reward po epizodi ===
plt.figure()
plt.plot(rewards)
plt.title("Reward po epizodi")
plt.xlabel("Epizoda")
plt.ylabel("Ukupan reward")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Vizualizacija 2: Agent vs Vozač ===
agent_pits, agent_comps, agent_styles = [], [], []
human_pits, human_comps, human_styles = [], [], []
obs, _ = env.reset(seed=123)

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    agent_pits.append(action[0])
    agent_comps.append(action[1])
    agent_styles.append(action[2])
    i = len(agent_pits) - 1
    if i < len(true_actions):
        hp, hc, hs = true_actions[i]
        human_pits.append(hp)
        human_comps.append(hc)
        human_styles.append(hs)
    if done:
        break

laps = list(range(1, len(agent_pits) + 1))

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(laps, agent_pits, label="Agent", marker='o')
plt.plot(laps, human_pits, label="Vozač", linestyle='--')
plt.ylabel("Pit stop")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(laps, agent_comps, label="Agent", marker='s')
plt.plot(laps, human_comps, label="Vozač", linestyle='--')
plt.ylabel("Gume (0–4)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(laps, agent_styles, label="Agent", marker='^')
plt.plot(laps, human_styles, label="Vozač", linestyle='--')
plt.ylabel("Stil (0–2)")
plt.xlabel("Krug")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# === Vizualizacija 3: Reward po vremenskim uvjetima ===
labels = list(weather_rewards.keys())
data = [np.mean(weather_rewards[w]) if len(weather_rewards[w]) > 0 else 0 for w in labels]

plt.figure()
plt.bar(labels, data)
plt.title("Prosječan reward po vremenskim uvjetima")
plt.ylabel("Reward")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# === Vizualizacija 4: Distribucija guma ===
compound_names = {0: "SOFT", 1: "MEDIUM", 2: "HARD", 3: "INTER", 4: "WET"}
counts = Counter([a[1] for a in true_actions])
labels = [compound_names.get(k, str(k)) for k in counts]
values = [counts[k] for k in counts]

plt.figure()
plt.bar(labels, values)
plt.title("Distribucija guma korištenih u podacima")
plt.ylabel("Broj korištenja")
plt.tight_layout()
plt.show()


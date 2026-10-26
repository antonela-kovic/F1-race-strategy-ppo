# train.py
# Trains the PPO agent with a three-level curriculum (easy -> medium -> full),
# saves the best and the final model, and plots training diagnostics.
from stable_baselines3 import PPO
from f1_env import FastF1MultiActionEnv
import matplotlib.pyplot as plt
from utils import selected_features
import numpy as np
from collections import Counter

obs_array = np.load("data/obs_array.npy")
true_actions = np.load("data/true_actions.npy")

env = FastF1MultiActionEnv(obs_array, true_actions=true_actions, difficulty_level="easy")

# Standard MLP policy with three hidden layers (256-128-64, ReLU, the SB3
# MlpPolicy default). The LSTM feature extractor in model.py is intentionally
# not used (see the note in model.py).
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[256, 128, 64]),
    ent_coef=0.02,     # entropy bonus to keep exploring all three action heads
    verbose=1
)

rewards = []
weather_rewards = {"dry": [], "damp": [], "rain": []}

# Counters of real vs synthetically injected laps (see _augment_scenarios in
# f1_env.py), reported per curriculum level to document how much of the
# training data was real dry laps vs synthetic rain/damp and SC/VSC scenarios.
cum_total_laps = 0
cum_synth_weather_laps = 0
cum_synth_caution_laps = 0
per_level_stats = {}  # {"easy": {"total":..,"synth_weather":..,"synth_caution":..}, ...}

BEST_MODEL_PATH = "saved_models/ppo_f1_rl_logic_v6_best.zip"
best_avg_reward_per_step = -np.inf

for i in range(100):
    # === Curriculum ===
    if i in [30, 60]:
        env.increase_difficulty()
        print(f"Difficulty changed to: {env.difficulty_level}")

    model.learn(total_timesteps=2000, reset_num_timesteps=False)
    obs, _ = env.reset(seed=42)

    # === Weather distribution in this episode ===
    weather_map = {1: "dry", 2: "damp", 3: "rain"}
    weather_human = [weather_map.get(row[selected_features.index("Weather")], "?") for row in env.filtered_obs_array]
    print(f"Episode {i+1} - weather per lap:", weather_human[:10], "...")

    transitions = sum(weather_human[j] != weather_human[j-1] for j in range(1, len(weather_human)))
    print(f"Weather changes: {transitions}x")

    # === Real vs synthetic laps in this episode ===
    # env.is_synthetic_weather / env.is_synthetic_caution are boolean arrays
    # (set in f1_env.py -> _augment_scenarios) of the same length as
    # env.filtered_obs_array for the CURRENT episode.
    n_total = len(env.filtered_obs_array)
    n_synth_weather = int(env.is_synthetic_weather.sum())
    n_synth_caution = int(env.is_synthetic_caution.sum())
    n_synth_any = int((env.is_synthetic_weather | env.is_synthetic_caution).sum())
    n_real = n_total - n_synth_any
    print(
        f"Episode {i+1} - laps: {n_total} total | "
        f"real={n_real} ({n_real/n_total:.1%}) | "
        f"synthetic weather={n_synth_weather} ({n_synth_weather/n_total:.1%}) | "
        f"synthetic caution={n_synth_caution} ({n_synth_caution/n_total:.1%})"
    )

    cum_total_laps += n_total
    cum_synth_weather_laps += n_synth_weather
    cum_synth_caution_laps += n_synth_caution
    lvl = env.difficulty_level
    lvl_stats = per_level_stats.setdefault(lvl, {"total": 0, "synth_weather": 0, "synth_caution": 0, "episodes": 0})
    lvl_stats["total"] += n_total
    lvl_stats["synth_weather"] += n_synth_weather
    lvl_stats["synth_caution"] += n_synth_caution
    lvl_stats["episodes"] += 1

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
            print(f"Episode {i+1} finished after {steps} steps (reward = {total_reward:.2f})")
            break

    # === Smoothed reward ===
    smoothed = 0.95 * rewards[-1] + 0.05 * total_reward if rewards else total_reward
    rewards.append(smoothed)

    # === Save the BEST model, not only the last one ===
    # Uses reward PER STEP (not total reward) because episode length depends on
    # the agent's behaviour (an early DNF shortens the episode), so comparing
    # totals would favour longer episodes regardless of decision quality. PPO
    # can become unstable late in training; this keeps the best version.
    avg_reward_per_step = total_reward / steps
    if avg_reward_per_step > best_avg_reward_per_step:
        best_avg_reward_per_step = avg_reward_per_step
        model.save(BEST_MODEL_PATH)
        print(f"  New best model saved (reward/step={avg_reward_per_step:.2f}, episode {i+1})")


# === Summary: real vs synthetic laps over the whole training ===
print("\n" + "=" * 60)
print("SUMMARY - real vs synthetic laps over the whole training")
print("=" * 60)
cum_synth_any_approx = cum_synth_weather_laps + cum_synth_caution_laps  # upper bound (categories can overlap)
print(f"Total laps seen during training: {cum_total_laps}")
print(f"  - synthetic weather (rain/damp): {cum_synth_weather_laps} ({cum_synth_weather_laps/cum_total_laps:.1%})")
print(f"  - synthetic caution (SC/VSC/yellow): {cum_synth_caution_laps} ({cum_synth_caution_laps/cum_total_laps:.1%})")
print(f"  - remaining (mostly real dry laps): "
      f"{cum_total_laps - cum_synth_any_approx} (lower bound, because the two categories can overlap)")
print("\nPer curriculum level:")
for lvl, s in per_level_stats.items():
    print(
        f"  {lvl:>6} | episodes={s['episodes']:>3} | total laps={s['total']:>6} | "
        f"synth. weather={s['synth_weather']:>5} ({s['synth_weather']/s['total']:.1%}) | "
        f"synth. caution={s['synth_caution']:>5} ({s['synth_caution']/s['total']:.1%})"
    )
print("=" * 60 + "\n")


model.save("saved_models/ppo_f1_rl_logic_v6_retrained.zip")  # final model after all 100 episodes
print(f"Final model saved: saved_models/ppo_f1_rl_logic_v6_retrained.zip")
print(f"Best model (reward/step = {best_avg_reward_per_step:.2f}) saved: {BEST_MODEL_PATH}")
print("If training became unstable near the end (sudden reward drop), use BEST_MODEL_PATH "
      "in predict_strategies.py/api.py instead of the final model.")

# === Plot 1: reward per episode ===
plt.figure()
plt.plot(rewards)
plt.title("Reward per episode")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 2: agent vs driver ===
agent_pits, agent_comps, agent_styles = [], [], []
human_pits, human_comps, human_styles = [], [], []
obs, _ = env.reset(seed=123)

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    agent_pits.append(action[0])
    agent_comps.append(action[1])
    agent_styles.append(action[2])
    pos = len(agent_pits) - 1
    # The env shuffles the ORDER of whole races on every reset(), so position
    # `pos` in this episode is not the same as position `pos` in
    # true_actions.npy (always in the original order). env.orig_indices[pos]
    # gives the correct index.
    true_idx = env.orig_indices[pos]
    if true_idx < len(true_actions):
        hp, hc, hs = true_actions[true_idx]
        human_pits.append(hp)
        human_comps.append(hc)
        human_styles.append(hs)
    if done:
        break

laps = list(range(1, len(agent_pits) + 1))

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(laps, agent_pits, label="Agent", marker='o')
plt.plot(laps, human_pits, label="Driver", linestyle='--')
plt.ylabel("Pit stop")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(laps, agent_comps, label="Agent", marker='s')
plt.plot(laps, human_comps, label="Driver", linestyle='--')
plt.ylabel("Tyres (0-4)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(laps, agent_styles, label="Agent", marker='^')
plt.plot(laps, human_styles, label="Driver", linestyle='--')
plt.ylabel("Style (0-2)")
plt.xlabel("Lap")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# === Plot 3: reward per weather condition ===
labels = list(weather_rewards.keys())
data = [np.mean(weather_rewards[w]) if len(weather_rewards[w]) > 0 else 0 for w in labels]

plt.figure()
plt.bar(labels, data)
plt.title("Average reward per weather condition")
plt.ylabel("Reward")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# === Plot 4: tyre distribution ===
compound_names = {0: "SOFT", 1: "MEDIUM", 2: "HARD", 3: "INTER", 4: "WET"}
counts = Counter([a[1] for a in true_actions])
labels = [compound_names.get(k, str(k)) for k in counts]
values = [counts[k] for k in counts]

plt.figure()
plt.bar(labels, values)
plt.title("Distribution of tyres used in the data")
plt.ylabel("Number of uses")
plt.tight_layout()
plt.show()

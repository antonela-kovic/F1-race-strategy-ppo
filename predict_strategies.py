# predict_strategies.py
# NOTE: this evaluates on ALL laps, i.e. the same races the model was trained
# on (in-sample). For the generalisation estimate use evaluate_holdout.py.
# Runs the trained agent over all real laps, applies the safety layer and
# compares RAW (policy only) vs SAFE (policy + rules) accuracy against the
# real drivers' decisions. Also checks rule R10 per race.
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from evaluation import run_policy, compute_metrics, majority_class
from utils import STYLE_UNLABELED

# Model version history (only the last one is loaded):
#   v1 ppo_f1_rl.zip                    - no ent_coef, no safety layer
#   v2 ppo_f1_rl_logic_v2.zip           - added ent_coef and the safety layer
#   v3 ppo_f1_rl_logic_v3_finetuned.zip - fine-tuning for driving-style accuracy
#   v5 ppo_f1_rl_logic_v5_retrained.zip - fixed pit-stop reward in f1_env.py
#   v6 ppo_f1_rl_logic_v6_retrained.zip - scaled observations, LSTM removed,
#                                          per-race bookkeeping, asymmetric
#                                          pit reward (see f1_env.py)
#   v6_best (recommended) - checkpoint with the best reward/step during v6
#                           training, saved separately from the final model
#                           because PPO can become unstable late in training
model = PPO.load("saved_models/ppo_f1_rl_logic_v6_best.zip")
obs_array = np.load("data/obs_array.npy")
true_actions = np.load("data/true_actions.npy")

# run_policy() segments obs_array into races (a drop in LapNumber = new race or
# driver) and creates a new RaceState per race, so R10 is checked at the end of
# every race, not only at the end of the concatenated array.
raw_actions, safe_actions, stats = run_policy(model, obs_array)

print(f"[R10 check] Dry races in the dataset: {stats['dry_races']}")
print(f"[R10 check] Races satisfying the rule (>=2 dry compounds): "
      f"{stats['dry_races_r10_ok']}/{stats['dry_races']}")
print(f"[R10 check] Number of R10 interventions (across all races): {stats['r10_events']}")

# Choose what to save and plot:
predicted_actions = safe_actions   # default: safety layer enabled
# predicted_actions = raw_actions  # to see the behaviour without the safety layer

n = len(obs_array)
print(f"Safety layer override rate: {stats['laps_with_override']/n:.2%}")
print(f"Violations (raw): {stats['laps_with_violation']/n:.2%}")
print(f"Overrides by field: pit={stats['pit_overrides']}, "
      f"compound={stats['comp_overrides']}, style={stats['style_overrides']}")

df = pd.DataFrame(predicted_actions, columns=["Pit", "Compound", "Style"])
df.to_csv("data/predicted_strategies.csv", index=False)
print("Strategies saved to data/predicted_strategies.csv")

majority = (majority_class(true_actions[:, 0]), majority_class(true_actions[:, 1]),
            majority_class(true_actions[:, 2], ignore=STYLE_UNLABELED))
for name, pred in (("RAW = without the safety layer", raw_actions),
                   ("SAFE = with the safety layer", safe_actions)):
    m = compute_metrics(true_actions, pred, majority)
    print(f"\nPrediction accuracy ({name}, IN-SAMPLE):")
    print(f"Pit stop:      {m['pit_acc']:.2%} (recall {m['pit_recall']:.0%}, "
          f"precision {m['pit_precision']:.0%}, pit rate {m['pit_rate']:.1%})")
    print(f"Compound:      {m['compound_acc']:.2%} (baseline {m['baseline_compound_acc']:.2%})")
    print(f"Driving style: {m['style_acc']:.2%} on {m['n_style_labeled']} labeled laps "
          f"(baseline {m['baseline_style_acc']:.2%})")

# === Distribution plots ===
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(predicted_actions[:, 0], bins=np.arange(-0.5, 2), rwidth=0.6)
plt.title("Pit stop decisions")
plt.xticks([0, 1])
plt.xlabel("Pit (0/1)")

plt.subplot(1, 3, 2)
plt.hist(predicted_actions[:, 1], bins=np.arange(-0.5, 6), rwidth=0.6)
plt.title("Selected compounds")
plt.xlabel("Compound code")

plt.subplot(1, 3, 3)
plt.hist(predicted_actions[:, 2], bins=np.arange(-0.5, 4), rwidth=0.6)
plt.title("Driving style")
plt.xlabel("Style (0-2)")

plt.tight_layout()
plt.show()

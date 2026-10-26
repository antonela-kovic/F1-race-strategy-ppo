# predict_strategies.py
# Runs the trained agent over all real laps, applies the safety layer and
# compares RAW (policy only) vs SAFE (policy + rules) accuracy against the
# real drivers' decisions. Also checks rule R10 per race.
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import selected_features, scale_observation
from safety_layer import verify_action, RaceState

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

viol_before = 0
overrides_count = 0

raw_actions = []
safe_actions = []

# Safety-layer intervention counters
pit_overrides = 0
comp_overrides = 0
style_overrides = 0
mandatory_compound_events = 0  # how many times R10 (two mandatory compounds) fired

# IMPORTANT: obs_array is a CONCATENATION of several real races (different
# tracks/drivers), not one long race. Without segmentation, R10 ("last 3 laps
# before the end") would only check the last 3 rows of the whole array instead
# of the end of every race. Races are segmented by a drop in LapNumber
# (a drop means a new race or driver).
i_lap = selected_features.index("LapNumber")
i_weather = selected_features.index("Weather")

lap_numbers = obs_array[:, i_lap]
race_start_indices = [0]
for idx in range(1, len(lap_numbers)):
    if lap_numbers[idx] < lap_numbers[idx - 1]:
        race_start_indices.append(idx)
race_start_indices.append(len(obs_array))

n_races = len(race_start_indices) - 1
print(f"[Segmentation] Detected {n_races} separate races/stints in obs_array "
      f"(lengths: {[race_start_indices[k+1]-race_start_indices[k] for k in range(n_races)]})")

races_dry_ok = 0
races_dry_total = 0

race_state = None  # re-created for every race in the loop below
seg_end = 0

for row_idx, obs in enumerate(obs_array):
    # First lap of a new race? Then create a NEW RaceState.
    if row_idx in race_start_indices[:-1]:
        pos = race_start_indices.index(row_idx)
        seg_end = race_start_indices[pos + 1]
        race_state = RaceState(total_laps=seg_end - row_idx, force_window=3)

    action, _ = model.predict(scale_observation(obs).reshape(1, -1), deterministic=True)
    raw = action[0] if isinstance(action[0], (list, np.ndarray)) else action
    raw = np.array(raw, dtype=np.int32)

    raw_actions.append(raw)

    # verify_action receives the RAW (unscaled) state, because its thresholds
    # (tyre_life < 2.0, front_gap < 1.5, etc.) are defined in real units.
    safe, report = verify_action(obs, raw, selected_features, race_state=race_state)

    # Count overrides per field (pit/compound/style)
    for o in report.overrides:
        if o["field"] == "pit":
            pit_overrides += 1
        elif o["field"] == "compound":
            comp_overrides += 1
        elif o["field"] == "style":
            style_overrides += 1
        if o["rule"] == "R10_MANDATORY_TWO_DRY_COMPOUNDS":
            mandatory_compound_events += 1

    # metrics
    if len(report.violations_before) > 0:
        viol_before += 1
    if len(report.overrides) > 0:
        overrides_count += 1

    safe_actions.append(safe)

    # Update THIS race's history with the SAFE (applied) compound, not the raw one
    race_state.record(weather=int(round(float(obs[i_weather]))), compound=int(safe[1]))

    # Last lap of this race? Check whether the rule is satisfied.
    if row_idx + 1 == seg_end:
        if race_state.is_dry_race_so_far:
            races_dry_total += 1
            ok = len(race_state.dry_compounds_used) >= 2
            races_dry_ok += int(ok)

print(f"\n[R10 check] Dry races in the dataset: {races_dry_total}")
print(f"[R10 check] Races satisfying the rule (>=2 dry compounds): {races_dry_ok}/{races_dry_total}")
print(f"[R10 check] Number of R10 interventions (across all races): {mandatory_compound_events}")

raw_actions = np.array(raw_actions, dtype=np.int32)
safe_actions = np.array(safe_actions, dtype=np.int32)

# Choose what to save and plot:
predicted_actions = safe_actions   # default: safety layer enabled
# predicted_actions = raw_actions  # to see the behaviour without the safety layer

n = len(obs_array)
print(f"Safety layer override rate: {overrides_count/n:.2%}" if n else "No samples.")
print(f"Violations (raw): {viol_before/len(obs_array):.2%}")
print(f"Overrides by field: pit={pit_overrides}, compound={comp_overrides}, style={style_overrides}")

# Save as CSV
df = pd.DataFrame(predicted_actions, columns=["Pit", "Compound", "Style"])
df.to_csv("data/predicted_strategies.csv", index=False)
print("Strategies saved to data/predicted_strategies.csv")

# === Accuracy (RAW vs SAFE) ===
acc_pit_raw = accuracy_score(true_actions[:, 0], raw_actions[:, 0])
acc_comp_raw = accuracy_score(true_actions[:, 1], raw_actions[:, 1])
acc_style_raw = accuracy_score(true_actions[:, 2], raw_actions[:, 2])

acc_pit_safe = accuracy_score(true_actions[:, 0], safe_actions[:, 0])
acc_comp_safe = accuracy_score(true_actions[:, 1], safe_actions[:, 1])
acc_style_safe = accuracy_score(true_actions[:, 2], safe_actions[:, 2])

print("\nPrediction accuracy (RAW = without the safety layer):")
print(f"Pit stop:     {acc_pit_raw:.2%}")
print(f"Compound:     {acc_comp_raw:.2%}")
print(f"Driving style:{acc_style_raw:.2%}")

print("\nPrediction accuracy (SAFE = with the safety layer):")
print(f"Pit stop:     {acc_pit_safe:.2%}")
print(f"Compound:     {acc_comp_safe:.2%}")
print(f"Driving style:{acc_style_safe:.2%}")

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

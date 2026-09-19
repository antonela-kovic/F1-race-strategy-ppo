# === predict_strategies.py ===
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import selected_features
from logic_layer import verify_action

# Učitaj model i podatke ovisno o modelu koji ti treba
#model = PPO.load("saved_models/ppo_f1_rl.zip") --> stari model bez ent_coef u train.py i logic_layera
#model = PPO.load("saved_models/ppo_f1_rl_logic_v2.zip") # --> novi model sa ent_coef i logic_layer
#model = PPO.load("saved_models/ppo_f1_rl_logic_v3_finetuned.zip") #--> novi model sa finetuningom kako bi se povecala tocnost za stil
model = PPO.load("saved_models/ppo_f1_rl_logic_v4_retrained.zip") # model sa popravljenim pit stopom u f1_env.py
obs_array = np.load("data/obs_array.npy")
true_actions = np.load("data/true_actions.npy")


# Nadogradnja za logički sloj
viol_before = 0
#viol_after = 0
overrides_count = 0

raw_actions = [] # dodano zbog provejre narusavanja tocnosti pit stopa
safe_actions = []

# Brojaci intervencije logičkog sloja
pit_overrides = 0
comp_overrides = 0
style_overrides = 0

for obs in obs_array:
    action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
    raw = action[0] if isinstance(action[0], (list, np.ndarray)) else action
    raw = np.array(raw, dtype=np.int32)

    raw_actions.append(raw)   # <-- provjera tocnosti pit stop-a

    safe, report = verify_action(obs, raw, selected_features)

    # Broji override po polju (pit/compound/style)
    for o in report.overrides:
        if o["field"] == "pit":
            pit_overrides += 1
        elif o["field"] == "compound":
            comp_overrides += 1
        elif o["field"] == "style":
            style_overrides += 1
    

    # metrike
    if len(report.violations_before) > 0:
        viol_before += 1
    if len(report.overrides) > 0:
        overrides_count += 1
  

    safe_actions.append(safe)



# Izmjena zbog provjere pit stopa
raw_actions = np.array(raw_actions, dtype=np.int32)
safe_actions = np.array(safe_actions, dtype=np.int32)

# Odaberi što želiš spremati i prikazivati:
predicted_actions = safe_actions   # default: logički sloj uključen
# predicted_actions = raw_actions  # ako želiš vidjeti ponašanje bez logike

# Nadogradnja:  Dodaj guard da ne dijeliš s nulom (ako bi obs_array ikad bio prazan):

n = len(obs_array)
print(f"Logic layer override rate: {overrides_count/n:.2%}" if n else "No samples.")
print(f"Violations (raw): {viol_before/len(obs_array):.2%}")
# Ispis brojaca override-a po polju
print(f"Overrides by field: pit={pit_overrides}, compound={comp_overrides}, style={style_overrides}")

# Spremi kao CSV
df = pd.DataFrame(predicted_actions, columns=["Pit", "Compound", "Style"])
df.to_csv("data/predicted_strategies.csv", index=False)
print("Strategije spremljene u data/predicted_strategies.csv")

# === Evaluacija točnosti bez pit stopa ===
# acc_pit = accuracy_score(true_actions[:, 0], predicted_actions[:, 0])
# acc_comp = accuracy_score(true_actions[:, 1], predicted_actions[:, 1])
# acc_style = accuracy_score(true_actions[:, 2], predicted_actions[:, 2])

# print(f"Točnost predikcije:")
# print(f"Pit stop:     {acc_pit:.2%}")
# print(f"Compound:     {acc_comp:.2%}")
# print(f"Driving style:{acc_style:.2%}")

# === Evaluacija točnosti (RAW vs SAFE) ===
acc_pit_raw = accuracy_score(true_actions[:, 0], raw_actions[:, 0])
acc_comp_raw = accuracy_score(true_actions[:, 1], raw_actions[:, 1])
acc_style_raw = accuracy_score(true_actions[:, 2], raw_actions[:, 2])

acc_pit_safe = accuracy_score(true_actions[:, 0], safe_actions[:, 0])
acc_comp_safe = accuracy_score(true_actions[:, 1], safe_actions[:, 1])
acc_style_safe = accuracy_score(true_actions[:, 2], safe_actions[:, 2])

print("\nTočnost predikcije (RAW = bez logike):")
print(f"Pit stop:     {acc_pit_raw:.2%}")
print(f"Compound:     {acc_comp_raw:.2%}")
print(f"Driving style:{acc_style_raw:.2%}")

print("\nTočnost predikcije (SAFE = s logikom):")
print(f"Pit stop:     {acc_pit_safe:.2%}")
print(f"Compound:     {acc_comp_safe:.2%}")
print(f"Driving style:{acc_style_safe:.2%}")




# === Grafikon distribucije ===
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(predicted_actions[:, 0], bins=np.arange(-0.5, 2), rwidth=0.6)
plt.title("Pit stop odluke")
plt.xticks([0, 1])
plt.xlabel("Pit (0/1)")

plt.subplot(1, 3, 2)
plt.hist(predicted_actions[:, 1], bins=np.arange(-0.5, 6), rwidth=0.6)
plt.title("Odabrani compoundi")
plt.xlabel("Compound kod")

plt.subplot(1, 3, 3)
plt.hist(predicted_actions[:, 2], bins=np.arange(-0.5, 4), rwidth=0.6)
plt.title("Stil vožnje")
plt.xlabel("Stil (0-2)")

plt.tight_layout()
plt.show()


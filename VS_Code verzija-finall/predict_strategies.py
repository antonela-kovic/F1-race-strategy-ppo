# === predict_strategies.py ===
# === Nadogradnja 4: Predikcija strategija drugih vozača ===
# Dodajemo novu funkciju koja koristi trenirani agent da predvidi strategiju za stvarne lapove
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import selected_features, scale_observation
from logic_layer import verify_action, RaceState

# Učitaj model i podatke ovisno o modelu koji ti treba
# Povijest verzija modela (najnoviji dolje, samo taj se stvarno učitava):
#   v1 ppo_f1_rl.zip                    - bez ent_coef, bez logičkog sloja
#   v2 ppo_f1_rl_logic_v2.zip           - dodan ent_coef i logički sloj
#   v3 ppo_f1_rl_logic_v3_finetuned.zip - fine-tuning radi točnosti stila
#   v5 ppo_f1_rl_logic_v5_retrained.zip - popravljen pit stop u f1_env.py
#   v6 ppo_f1_rl_logic_v6_retrained.zip - skalirana opažanja, uklonjen LSTM,
#                                          per-utrka bookkeeping, asimetrična
#                                          nagrada za pit (vidi f1_env.py)
#   v6_best (preporučeno) - checkpoint s najboljim reward/korak tijekom
#                            treninga v6, spremljen odvojeno od zadnjeg modela
#                            jer PPO zna biti nestabilan pred kraj treninga
model = PPO.load("saved_models/ppo_f1_rl_logic_v6_best.zip")
obs_array = np.load("data/obs_array.npy")
true_actions = np.load("data/true_actions.npy")

# Predikcije
# predicted_actions = []
# for obs in obs_array:
#     action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
#     action = action[0] if isinstance(action[0], (list, np.ndarray)) else action
#     predicted_actions.append(action)

# predicted_actions = np.array(predicted_actions)
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
mandatory_compound_events = 0  # koliko puta je aktivirano R10 (2 obavezna compounda)

# === VAŽNO: obs_array je SPOJ VIŠE STVARNIH UTRKA (različite staze/vozači), ===
# === NE jedna dugačka utrka! Bez segmentacije, R10 ("zadnja 3 kruga prije   ===
# === kraja") bi provjeravao samo zadnja 3 retka CIJELOG spojenog niza, a   ===
# === ne kraj svake pojedine utrke - pa bi se pravilo praktički nikad ne bi ===
# === aktiviralo za 11 od 12 utrka u datasetu.                              ===
# Segmentaciju radimo prema padu vrijednosti LapNumber (pad => nova utrka/vozač).
i_lap = selected_features.index("LapNumber")
i_weather = selected_features.index("Weather")

lap_numbers = obs_array[:, i_lap]
race_start_indices = [0]
for idx in range(1, len(lap_numbers)):
    if lap_numbers[idx] < lap_numbers[idx - 1]:
        race_start_indices.append(idx)
race_start_indices.append(len(obs_array))

n_races = len(race_start_indices) - 1
print(f"[Segmentacija] Detektirano {n_races} zasebnih utrka/stintova u obs_array "
      f"(duljine: {[race_start_indices[k+1]-race_start_indices[k] for k in range(n_races)]})")

races_dry_ok = 0
races_dry_total = 0

race_state = None  # instanciramo iznova za svaku utrku u petlji ispod
seg_end = 0

for row_idx, obs in enumerate(obs_array):
    # Je li ovo prvi krug nove utrke? Ako da, napravi NOVI RaceState.
    if row_idx in race_start_indices[:-1]:
        pos = race_start_indices.index(row_idx)
        seg_end = race_start_indices[pos + 1]
        race_state = RaceState(total_laps=seg_end - row_idx, force_window=3)

    action, _ = model.predict(scale_observation(obs).reshape(1, -1), deterministic=True)
    raw = action[0] if isinstance(action[0], (list, np.ndarray)) else action
    raw = np.array(raw, dtype=np.int32)

    raw_actions.append(raw)   # <-- DODAJ OVO za provjeru tocnosti pit stop-a

    # verify_action i dalje prima SIROVO (neskalirano) stanje - njegovi pragovi
    # (tyre_life < 2.0, front_gap < 1.5, itd.) pretpostavljaju stvarne jedinice.
    safe, report = verify_action(obs, raw, selected_features, race_state=race_state)

    # Broji override po polju (pit/compound/style)
    for o in report.overrides:
        if o["field"] == "pit":
            pit_overrides += 1
        elif o["field"] == "compound":
            comp_overrides += 1
        elif o["field"] == "style":
            style_overrides += 1
        if o["rule"] == "R10_MANDATORY_TWO_DRY_COMPOUNDS":
            mandatory_compound_events += 1

    # metrike
    if len(report.violations_before) > 0:
        viol_before += 1
    if len(report.overrides) > 0:
        overrides_count += 1

    safe_actions.append(safe)

    # Ažuriraj povijest TE utrke SA SAFE (primijenjenim) compoundom, ne raw-om
    race_state.record(weather=int(round(float(obs[i_weather]))), compound=int(safe[1]))

    # Je li ovo zadnji krug ove utrke? Ako da, provjeri je li pravilo zadovoljeno.
    if row_idx + 1 == seg_end:
        if race_state.is_dry_race_so_far:
            races_dry_total += 1
            ok = len(race_state.dry_compounds_used) >= 2
            races_dry_ok += int(ok)

print(f"\n[R10 provjera] Suhih utrka u datasetu: {races_dry_total}")
print(f"[R10 provjera] Od toga zadovoljavaju pravilo (≥2 suha compounda): {races_dry_ok}/{races_dry_total}")
print(f"[R10 provjera] Broj puta kad je R10 morao intervenirati (ukupno kroz sve utrke): {mandatory_compound_events}")


# safe_actions = np.array(safe_actions, dtype=np.int32)
# predicted_actions = np.array(safe_actions)
# Izmjena zbog provjere pit stopa
raw_actions = np.array(raw_actions, dtype=np.int32)
safe_actions = np.array(safe_actions, dtype=np.int32)

# Odaberi što želiš spremati i prikazivati:
predicted_actions = safe_actions   # default: logički sloj uključen
# predicted_actions = raw_actions  # ako želiš vidjeti ponašanje bez logike

# Nadogradnja:  Dodaj guard da ne dijeliš s nulom (ako bi obs_array ikad bio prazan):
# print(f"Logic layer override rate: {overrides_count/len(obs_array):.2%}")
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

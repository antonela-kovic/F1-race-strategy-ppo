# === train.py ===
from stable_baselines3 import PPO
from f1_env import FastF1MultiActionEnv
import matplotlib.pyplot as plt
from utils import selected_features
import numpy as np
from collections import Counter

obs_array = np.load("data/obs_array.npy")
true_actions = np.load("data/true_actions.npy")

env = FastF1MultiActionEnv(obs_array, true_actions=true_actions, difficulty_level="easy")

# NAPOMENA: ranije je ovdje bio korišten LSTMFeatureExtractor (model.py), no
# on je bio pogrešno implementiran - svaki forward() poziv gradi NOVU
# "sekvencu" duljine 1 s nultim skrivenim stanjem (obs.unsqueeze(1)), pa
# nikad nije stvarno pamtio prethodne krugove (za to bi trebao RecurrentPPO
# iz sb3-contrib, koji ovdje NIJE korišten). Efektivno je to bio samo
# nepotreban, nasumično inicijaliziran filter između sirovih značajki i
# politike - dodavao je šum bez ikakve koristi, i bio u suprotnosti s onim
# što rad tvrdi ("trenutna verzija ne koristi memorijske slojeve"). Uklonjen
# je; net_arch niže odgovara arhitekturi opisanoj u radu (tri skrivena sloja
# 256-128-64, ReLU aktivacija - SB3 MlpPolicy koristi ReLU po defaultu).
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[256, 128, 64]),
    ent_coef=0.02,     # <-- OVDJE
    verbose=1
)

rewards = []
weather_rewards = {"suho": [], "mokro": [], "kiša": []}

# Brojači stvarnih vs sintetički ubačenih krugova (vidi f1_env.py: _augment_scenarios)
# - korisno da se u radu točno navede koliko je agent tijekom treninga stvarno
# vidio suhih/stvarnih krugova naspram sintetički umetnutih scenarija kiše/
# vlage i Safety Car/VSC, po razini težine (curriculum).
cum_total_laps = 0
cum_synth_weather_laps = 0
cum_synth_caution_laps = 0
per_level_stats = {}  # {"easy": {"total":..,"synth_weather":..,"synth_caution":..}, ...}

BEST_MODEL_PATH = "saved_models/ppo_f1_rl_logic_v6_best.zip"
best_avg_reward_per_step = -np.inf

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

    # === Stvarni vs sintetički krugovi u ovoj epizodi ===
    # env.is_synthetic_weather / env.is_synthetic_caution su boolean nizovi
    # (postavljeni u f1_env.py -> _augment_scenarios) iste duljine kao
    # env.filtered_obs_array za TRENUTNU epizodu.
    n_total = len(env.filtered_obs_array)
    n_synth_weather = int(env.is_synthetic_weather.sum())
    n_synth_caution = int(env.is_synthetic_caution.sum())
    n_synth_any = int((env.is_synthetic_weather | env.is_synthetic_caution).sum())
    n_real = n_total - n_synth_any
    print(
        f"Epizoda {i+1} – Krugovi: {n_total} ukupno | "
        f"stvarni={n_real} ({n_real/n_total:.1%}) | "
        f"sintetičko vrijeme={n_synth_weather} ({n_synth_weather/n_total:.1%}) | "
        f"sintetički caution={n_synth_caution} ({n_synth_caution/n_total:.1%})"
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
            print(f"Epizoda {i+1} završila nakon {steps} koraka (reward = {total_reward:.2f})")
            break

    # === Smoothed reward ===
    smoothed = 0.95 * rewards[-1] + 0.05 * total_reward if rewards else total_reward
    rewards.append(smoothed)

    # === Spremi NAJBOLJI model, ne samo zadnji ===
    # Koristimo reward PO KORAKU (ne ukupan reward) jer duljina epizode sad
    # ovisi o ponašanju agenta (rani DNF kod rizičnog stila u lošem vremenu
    # skraćuje epizodu) - usporedba ukupnog reward-a bi favorizirala dulje
    # epizode neovisno o kvaliteti odluka. PPO zna biti nestabilan pred kraj
    # treninga (npr. nagli pad reward-a u kasnijim epizodama) - ovako se
    # najbolja verzija ne izgubi ako se to dogodi.
    avg_reward_per_step = total_reward / steps
    if avg_reward_per_step > best_avg_reward_per_step:
        best_avg_reward_per_step = avg_reward_per_step
        model.save(BEST_MODEL_PATH)
        print(f"  Novi najbolji model spremljen (reward/korak={avg_reward_per_step:.2f}, epizoda {i+1})")


# === Sažetak: stvarni vs sintetički krugovi kroz cijeli trening ===
print("\n" + "=" * 60)
print("SAŽETAK - stvarni vs sintetički krugovi kroz cijeli trening")
print("=" * 60)
cum_synth_any_approx = cum_synth_weather_laps + cum_synth_caution_laps  # gornja procjena (mogu se preklapati)
print(f"Ukupno krugova viđenih tijekom treninga: {cum_total_laps}")
print(f"  - sintetičko vrijeme (kiša/vlaga): {cum_synth_weather_laps} ({cum_synth_weather_laps/cum_total_laps:.1%})")
print(f"  - sintetički caution (SC/VSC/žuti): {cum_synth_caution_laps} ({cum_synth_caution_laps/cum_total_laps:.1%})")
print(f"  - preostalo (uglavnom stvarni suhi krugovi): "
      f"{cum_total_laps - cum_synth_any_approx} (donja procjena, jer se dvije kategorije mogu preklapati)")
print("\nPo razini težine (curriculum):")
for lvl, s in per_level_stats.items():
    print(
        f"  {lvl:>6} | epizoda={s['episodes']:>3} | ukupno krugova={s['total']:>6} | "
        f"sint. vrijeme={s['synth_weather']:>5} ({s['synth_weather']/s['total']:.1%}) | "
        f"sint. caution={s['synth_caution']:>5} ({s['synth_caution']/s['total']:.1%})"
    )
print("=" * 60 + "\n")


model.save("saved_models/ppo_f1_rl_logic_v6_retrained.zip") # zadnji model nakon svih 100 epizoda
print(f"Zadnji model spremljen: saved_models/ppo_f1_rl_logic_v6_retrained.zip")
print(f"Najbolji model (po reward/korak = {best_avg_reward_per_step:.2f}) spremljen: {BEST_MODEL_PATH}")
print("Ako je trening pred kraj bio nestabilan (nagli pad reward-a), koristi BEST_MODEL_PATH "
      "u predict_strategies.py/api.py umjesto zadnjeg modela.")
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
    pos = len(agent_pits) - 1
    # VAŽNO: env sad miješa REDOSLIJED CIJELIH UTRKA po svakom reset()-u
    # (f1_env.py -> _shuffle_race_order), pa pozicija `pos` u ovoj epizodi
    # više NIJE ista kao pozicija `pos` u true_actions.npy (koji je uvijek
    # u IZVORNOM, neizmiješanom redoslijedu). Bez env.orig_indices ovdje bismo
    # (kao ranije) uspoređivali agentovu odluku s odlukom vozača iz posve
    # DRUGE utrke/kruga - env.orig_indices[pos] daje ispravan indeks.
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

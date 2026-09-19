# finetune.py
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

from f1_env import FastF1MultiActionEnv
from utils import selected_features  # nije nužno ovdje, ali neka stoji konzistentno

def main():
    # --- data ---
    obs_array = np.load("data/obs_array.npy")
    true_actions = np.load("data/true_actions.npy")

    env = FastF1MultiActionEnv(obs_array, true_actions=true_actions, difficulty_level="easy")

    Path("saved_models").mkdir(parents=True, exist_ok=True)

    BASE = "saved_models/ppo_f1_rl_logic_v2.zip"
    OUT  = "saved_models/ppo_f1_rl_logic_v3_finetuned.zip"

    if not Path(BASE).exists():
        raise FileNotFoundError(f"Base model not found: {BASE}")

    print(f"Fine-tune: učitavam model: {BASE}")
    model = PPO.load(BASE, env=env)

    # --- fine-tune ---
    # Ovo je “sigurna” količina za vidjeti pomak u style bez rušenja compound-a.
    total_timesteps = 300_000

    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    model.save(OUT)
    print(f"Fine-tuned model spremljen u: {OUT}")

if __name__ == "__main__":

    main()

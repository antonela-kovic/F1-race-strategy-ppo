# === api.py ===
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from stable_baselines3 import PPO
from typing import Optional
from utils import selected_features, scale_observation
from logic_layer import verify_action, humanize_action, RaceState
from collections import Counter

# Učitaj trenirani model
# Vidi predict_strategies.py za povijest verzija modela. v6_best je
# checkpoint s najboljim reward/korak tijekom treninga (preporučeno umjesto
# zadnjeg modela, koji zna biti degradiran zbog PPO nestabilnosti pred kraj).
model = PPO.load("saved_models/ppo_f1_rl_logic_v6_best.zip")

# Sigurniji pristup ucitanju treniranog modela
# from pathlib import Path
# MODEL_PATH = Path(__file__).resolve().parent / "saved_models" / "ppo_f1_rl.zip"
# model = PPO.load(str(MODEL_PATH))

# Pokreni FastAPI aplikaciju
app = FastAPI()

# === Pydantic modeli ===
class Observation(BaseModel):
    state: List[float]
    # Opcionalni kontekst utrke - REST poziv je bez stanja (stateless), pa
    # pravilo "obavezna 2 tipa suhih guma" (R10) treba da mu KLIJENT (npr.
    # main.ipynb ili predict_strategies.py) proslijedi dosadašnju povijest
    # ove utrke. Bez ovoga R10 se jednostavno preskače za taj poziv, ostala
    # pravila (R1-R9) rade normalno jer se oslanjaju samo na trenutni state.
    compounds_used: Optional[List[int]] = None
    wet_or_damp_seen: Optional[bool] = None
    laps_remaining: Optional[int] = None

class ActionOutput(BaseModel):
    action: List[int]
    explanation: dict

class BatchOutput(BaseModel):
    actions: List[List[int]]

# === Pomoćna funkcija: sastavi RaceState iz opcionalnog konteksta u zahtjevu ===
def _build_race_state(obs: "Observation") -> Optional[RaceState]:
    if obs.laps_remaining is None:
        return None  # klijent nije poslao kontekst -> R10 se preskače za ovaj poziv
    rs = RaceState(total_laps=obs.laps_remaining + 1, force_window=3)
    rs.compounds_used = set(obs.compounds_used or [])
    rs.wet_or_damp_seen = bool(obs.wet_or_damp_seen)
    return rs


# === Korijen aplikacije ===
@app.get("/")
def root():
    return {"message": "F1 RL Strategy API is running."}

# === Jedan primjer predikcije ===
@app.post("/predict_strategy", response_model=ActionOutput)
def predict_strategy(obs: Observation):
    if len(obs.state) != len(selected_features):
        raise HTTPException(
            status_code=400,
            detail=f"Očekivano {len(selected_features)} značajki, dobijeno {len(obs.state)}"
        )

    state = np.array([obs.state], dtype=np.float32)
    action, _ = model.predict(scale_observation(state), deterministic=True)

    # Logički sloj - nadogradnja
    raw_action = action[0] if isinstance(action[0], (list, np.ndarray)) else action
    raw_action = np.array(raw_action, dtype=np.int32)

    state_1d = state.reshape(-1)  # (n_features,) - SIROVO, za verify_action
    race_state = _build_race_state(obs)
    safe_action, report = verify_action(state_1d, raw_action, selected_features, race_state=race_state)

    explanation = {
        "raw_action": raw_action.tolist(),
        "raw_human": humanize_action(raw_action),
        "safe_action": safe_action.tolist(),
        "safe_human": humanize_action(safe_action),
        "logic": {
            "violations_before": report.violations_before,
            "overrides": report.overrides,
            "rules_triggered": report.rules_triggered
        }
    }

    return {"action": safe_action.tolist(), "explanation": explanation}

# === Više primjera (batch predikcija) ===
@app.post("/predict_strategy_batch")
def predict_strategy_batch(batch: list[Observation]):
    if not batch:
        return {
            "summary": {"n": 0, "override_rate": 0.0, "violation_rate_raw": 0.0, "top_rules": []},
            "results": []
        }

    results = []
    overrides_count = 0
    violations_count = 0
    rules_counter = Counter()

    # Batch se tretira kao JEDNA sekvencijalna utrka (redoslijed iz liste = redoslijed
    # krugova), pa R10 (obavezna 2 compounda) ovdje MOŽE raditi automatski bez
    # dodatnog konteksta od klijenta - za razliku od pojedinačnog /predict_strategy poziva.
    i_weather = selected_features.index("Weather")
    race_state = RaceState(total_laps=len(batch), force_window=3)

    for obs in batch:
        if len(obs.state) != len(selected_features):
            raise HTTPException(
                status_code=400,
                detail=f"Očekivano {len(selected_features)} značajki, dobijeno {len(obs.state)}"
            )

        state = np.array([obs.state], dtype=np.float32)
        action, _ = model.predict(scale_observation(state), deterministic=True)

        raw_action = action[0] if isinstance(action[0], (list, np.ndarray)) else action
        raw_action = np.array(raw_action, dtype=np.int32)

        safe_action, report = verify_action(state.reshape(-1), raw_action, selected_features, race_state=race_state)
        race_state.record(weather=int(round(float(obs.state[i_weather]))), compound=int(safe_action[1]))

        if report.violations_before:
            violations_count += 1
        if report.overrides:
            overrides_count += 1
        rules_counter.update(report.rules_triggered)

        results.append({
            "raw_action": raw_action.tolist(),
            "raw_human": humanize_action(raw_action),
            "safe_action": safe_action.tolist(),
            "safe_human": humanize_action(safe_action),
            "logic": {
                "violations_before": report.violations_before,
                "overrides": report.overrides,
                "rules_triggered": report.rules_triggered
            }
        })

    n = len(batch)
    summary = {
        "n": n,
        "override_rate": overrides_count / n,
        "violation_rate_raw": violations_count / n,
        "top_rules": rules_counter.most_common(10),
        "mandatory_two_compounds": {
            "dry_race": race_state.is_dry_race_so_far,
            "distinct_dry_compounds_used": sorted(race_state.dry_compounds_used),
            "rule_satisfied": (not race_state.is_dry_race_so_far)
                              or len(race_state.dry_compounds_used) >= 2,
        }
    }

    return {"summary": summary, "results": results}

# === Objašnjenje strategije bez modela ===
@app.post("/explain_strategy")
def explain_strategy(obs: Observation):
    if len(obs.state) != len(selected_features):
        raise HTTPException(
            status_code=400,
            detail=f"Očekivano {len(selected_features)} značajki, dobijeno {len(obs.state)}"
        )

    state = np.array(obs.state, dtype=np.float32)
    i = lambda name: selected_features.index(name)

    tyre_life = state[i("TyreLife")]
    compound = int(state[i("CompoundEncoded")])
    weather_temp = state[i("TrackTemperature")]
    front_gap = state[i("FrontGap")]

    analysis = {}

    if tyre_life < 8:
        analysis["pit"] = "Preporučen PIT - niska izdržljivost guma."
    else:
        analysis["pit"] = "PIT nije potreban - gume još dobre."

    if weather_temp < 20:
        analysis["compound"] = "Niska temperatura staze - tvrđe gume bolje drže."
    elif weather_temp > 32:
        analysis["compound"] = "Vruća staza - soft gume brže degradiraju."
    else:
        analysis["compound"] = "Uvjeti srednji - MEDIUM bi mogao biti optimalan."

    if front_gap < 1.5:
        analysis["style"] = "Manji razmak - konzervativan stil za izbjegavanje sudara."
    else:
        analysis["style"] = "Prazniji prostor - možeš biti agresivniji."

    return {"explanation": analysis}

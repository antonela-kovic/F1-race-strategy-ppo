# api.py
# FastAPI service that returns race-strategy predictions from the trained PPO
# agent, verified and explained by the rule-based safety layer.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from stable_baselines3 import PPO
from typing import Optional
from utils import selected_features, scale_observation
from safety_layer import verify_action, humanize_action, RaceState
from collections import Counter

# Load the trained model. v6_best is the checkpoint with the best reward/step
# during training (recommended over the final model, which can degrade due to
# PPO instability late in training). See predict_strategies.py for the version history.
model = PPO.load("saved_models/ppo_f1_rl_logic_v6_best.zip")

app = FastAPI(title="F1 RL Strategy API")

# === Pydantic models ===
class Observation(BaseModel):
    state: List[float]
    # Optional race context. The REST call is stateless, so for rule R10
    # (two mandatory dry compounds) the CLIENT must pass the race history so
    # far. Without it, R10 is skipped for that call; rules R1-R9 work as usual
    # because they rely only on the current state.
    compounds_used: Optional[List[int]] = None
    wet_or_damp_seen: Optional[bool] = None
    laps_remaining: Optional[int] = None

class ActionOutput(BaseModel):
    action: List[int]
    explanation: dict

class BatchOutput(BaseModel):
    actions: List[List[int]]

# === Helper: build a RaceState from the optional context in the request ===
def _build_race_state(obs: "Observation") -> Optional[RaceState]:
    if obs.laps_remaining is None:
        return None  # no context from the client -> R10 is skipped for this call
    rs = RaceState(total_laps=obs.laps_remaining + 1, force_window=3)
    rs.compounds_used = set(obs.compounds_used or [])
    rs.wet_or_damp_seen = bool(obs.wet_or_damp_seen)
    return rs


# === Root ===
@app.get("/")
def root():
    return {"message": "F1 RL Strategy API is running."}

# === Single prediction ===
@app.post("/predict_strategy", response_model=ActionOutput)
def predict_strategy(obs: Observation):
    if len(obs.state) != len(selected_features):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(selected_features)} features, got {len(obs.state)}"
        )

    state = np.array([obs.state], dtype=np.float32)
    action, _ = model.predict(scale_observation(state), deterministic=True)

    raw_action = action[0] if isinstance(action[0], (list, np.ndarray)) else action
    raw_action = np.array(raw_action, dtype=np.int32)

    state_1d = state.reshape(-1)  # (n_features,) - RAW (unscaled), for verify_action
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

# === Batch prediction ===
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

    # The batch is treated as ONE sequential race (list order = lap order), so
    # R10 (two mandatory compounds) works automatically without extra context
    # from the client, unlike the single /predict_strategy call.
    i_weather = selected_features.index("Weather")
    race_state = RaceState(total_laps=len(batch), force_window=3)

    for obs in batch:
        if len(obs.state) != len(selected_features):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(selected_features)} features, got {len(obs.state)}"
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

# === Rule-of-thumb explanation without the model ===
@app.post("/explain_strategy")
def explain_strategy(obs: Observation):
    if len(obs.state) != len(selected_features):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(selected_features)} features, got {len(obs.state)}"
        )

    state = np.array(obs.state, dtype=np.float32)
    i = lambda name: selected_features.index(name)

    tyre_life = state[i("TyreLife")]
    compound = int(state[i("CompoundEncoded")])
    weather_temp = state[i("TrackTemperature")]
    front_gap = state[i("FrontGap")]

    analysis = {}

    if tyre_life < 8:
        analysis["pit"] = "PIT recommended - low tyre life."
    else:
        analysis["pit"] = "No PIT needed - tyres are still good."

    if weather_temp < 20:
        analysis["compound"] = "Low track temperature - harder tyres hold better."
    elif weather_temp > 32:
        analysis["compound"] = "Hot track - soft tyres degrade faster."
    else:
        analysis["compound"] = "Moderate conditions - MEDIUM may be optimal."

    if front_gap < 1.5:
        analysis["style"] = "Small gap - conservative style to avoid a collision."
    else:
        analysis["style"] = "More free space - a more aggressive style is possible."

    return {"explanation": analysis}

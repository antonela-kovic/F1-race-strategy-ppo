# utils.py
# Shared feature definitions, observation scaling and pure helper functions.
import numpy as np

selected_features = [
    "TyreLife", "CompoundEncoded", "LapNumber", "TrackStatus",
    "IsAccurate", "Position", "LapTime", "Sector1Time", "Sector2Time",
    "FrontGap", "RearGap", "Stint", "TrackTemperature", "AirTemperature",
    "Weather"  # real weather code derived from FastF1 data in prepare_data.py
]


def load_obs_array(path="data/obs_array.npy"):
    return np.load(path)

def load_true_actions(path="data/true_actions.npy"):
    return np.load(path)


# === Observation scaling (normalisation) ===
# Features have very different ranges (e.g. LapTime ~80-150 vs FrontGap
# 0.5-5). Without normalisation the network effectively ignores small-range
# features: the policy learned compound choice (driven by TyreLife and
# TrackTemperature, larger ranges) but not driving style, which depends almost
# entirely on FrontGap/RearGap.
#
# Use THIS function everywhere an observation is passed to the model:
# f1_env.py (training), predict_strategies.py and api.py (inference). A model
# trained on scaled observations gives meaningless predictions on unscaled
# ones and vice versa. verify_action() in safety_layer.py still receives the
# RAW (unscaled) state, because its thresholds (e.g. tyre_life < 2.0,
# front_gap < 1.5) are defined in real units.
FEATURE_SCALE = {
    "TyreLife": 40.0,
    "CompoundEncoded": 4.0,
    "LapNumber": 70.0,
    "TrackStatus": 7.0,
    "IsAccurate": 1.0,
    "Position": 20.0,
    "LapTime": 100.0,
    "Sector1Time": 40.0,
    "Sector2Time": 40.0,
    "FrontGap": 5.0,
    "RearGap": 5.0,
    "Stint": 5.0,
    "TrackTemperature": 45.0,
    "AirTemperature": 35.0,
    "Weather": 3.0,
}
_SCALE_VECTOR = np.array([FEATURE_SCALE[f] for f in selected_features], dtype=np.float32)


def scale_observation(obs):
    """
    Return the SCALED observation: every feature is divided by its scale from
    FEATURE_SCALE (in the order of selected_features), so all features end up
    in a similar range (~0-1.5). Works on a single row (shape (15,)) and on a
    2D array of rows (shape (N, 15)).
    """
    arr = np.asarray(obs, dtype=np.float32)
    return arr / _SCALE_VECTOR


# === Pure helper functions for data preparation (prepare_data.py) ===
# These live here instead of in prepare_data.py because prepare_data.py starts
# FastF1 network calls at import time. They are plain Python (no pandas or
# fastf1 dependency), so they can be unit-tested in isolation.

# Same convention as the agent's action space (0 SOFT, 1 MEDIUM, 2 HARD,
# 3 INTERMEDIATE, 4 WET), not the arbitrary alphabetical order of LabelEncoder.
COMPOUND_TO_CODE = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}


def classify_weather(rainfall_series, damp_memory_laps=3):
    """
    Convert a sequence of boolean Rainfall values (one per lap, in
    chronological order within ONE race/driver) into Weather codes:
      3 = rain (Rainfall active on that lap)
      2 = damp (rain stopped recently and the track is still wet; lasts
          `damp_memory_laps` laps after the last rainy lap)
      1 = dry (it never rained, or the rain stopped long ago)
    """
    weather = []
    damp_left = 0
    for is_rain in rainfall_series:
        if bool(is_rain):
            weather.append(3)
            damp_left = damp_memory_laps
        elif damp_left > 0:
            weather.append(2)
            damp_left -= 1
        else:
            weather.append(1)
    return weather


# FastF1 stores TrackStatus as a STRING of concatenated codes that occurred
# during the lap (e.g. "41" means statuses 4 AND 1 occurred, "1264" means
# 1, 2, 6, 4; they are not the numbers 41 or 1264). Priority for decoding into
# ONE most severe status: 5=red (most severe), then SC, VSC, VSC ending,
# yellow, clear.
STATUS_PRIORITY = [5, 4, 6, 7, 2, 1]


def worst_track_status(raw) -> int:
    """Decode FastF1's concatenated TrackStatus string into a single (most
    severe) status code. Returns 1 (clear) for unknown or empty input."""
    digits = set(str(raw))
    for code in STATUS_PRIORITY:
        if str(code) in digits:
            return code
    return 1

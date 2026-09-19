# === utils.py ===
import numpy as np

# === utils.py ===
selected_features = [
    "TyreLife", "CompoundEncoded", "LapNumber", "TrackStatus",
    "IsAccurate", "Position", "LapTime", "Sector1Time", "Sector2Time",
    "FrontGap", "RearGap", "Stint", "TrackTemperature", "AirTemperature",
    "Weather"  # dodajemo jer se Weather sada gleda kao zančajka i ne uzima se iz funkcije iznad
]


# Ako pokrecemo test_env.py za brzo testiranje
def load_obs_array(path="data/obs_array.npy"):
    return np.load(path)

def load_true_actions(path="data/true_actions.npy"):
    return np.load(path)


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


# === Real gap features (FrontGap / RearGap) ===
# FastF1 has no per-lap "gap" column, but every lap of EVERY driver has `Time`,
# the session time at which the car crossed the timing line at the end of that
# lap, and `Position` after that lap. The gap to the car ahead at the end of lap
# N is therefore the difference between the two crossing times on lap N. This
# is the same definition the official timing screens use for "interval".
# The function must receive the laps of ALL drivers in the session, not only the
# drivers used for training, otherwise the car ahead is usually missing.
GAP_CAP_SECONDS = 10.0  # leader / last car / missing neighbour; also keeps
                        # FrontGap / 5.0 inside the observation bound of 2.0


def compute_gaps(laps, cap=GAP_CAP_SECONDS):
    """
    Return a copy of `laps` (pandas DataFrame with columns Driver, LapNumber,
    Position and Time; Time as timedelta or seconds) with two new columns:
      FrontGap - seconds to the car one position ahead on the same lap
      RearGap  - seconds to the car one position behind on the same lap
    Values are clipped to [0, cap]; the leader gets FrontGap=cap, the last
    classified car RearGap=cap, and laps without a Position/Time get cap.
    """
    import pandas as pd

    df = laps.copy()
    t = df["Time"]
    t_sec = t.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(t) else t.astype(float)
    df["_t"] = t_sec
    df["FrontGap"] = float(cap)
    df["RearGap"] = float(cap)

    valid = df["Position"].notna() & df["_t"].notna()
    for _, g in df[valid].groupby("LapNumber"):
        g = g.sort_values("Position")
        times = g["_t"].to_numpy()
        pos = g["Position"].to_numpy()
        front = np.full(len(g), float(cap))
        rear = np.full(len(g), float(cap))
        for k in range(1, len(g)):
            # only a true neighbour (position exactly one ahead) counts
            if pos[k] - pos[k - 1] == 1:
                gap = times[k] - times[k - 1]
                front[k] = gap
                rear[k - 1] = gap
        df.loc[g.index, "FrontGap"] = np.clip(front, 0.0, cap)
        df.loc[g.index, "RearGap"] = np.clip(rear, 0.0, cap)

    return df.drop(columns="_t")


# === Driving-style label from real pace ===
# F1 data has no "style" label. The previous label was a rule over FrontGap,
# RearGap and TyreLife, i.e. a deterministic function of features the agent
# also observes (and the gaps were random), so style accuracy only measured
# how well the network re-learned that rule.
#
# The label is now derived from MEASURED pace: within each stint of a driver,
# lap time is fitted as a linear function of tyre age (this captures tyre
# degradation and fuel burn, which are collinear inside a stint). The residual
# says how much faster or slower than his own stint trend the driver was on
# that lap:
#   residual < -threshold -> 2 aggressive (pushing harder than the trend)
#   residual > +threshold -> 0 conservative (managing / saving tyres)
#   otherwise             -> 1 neutral
# It is still a proxy (a slow lap can also be caused by traffic), but it comes
# from real timing data and is not a function of the observation alone.
#
# Laps where pace says nothing about the driver's choice (lap 1, pit in/out
# laps, Safety Car/VSC/yellow, laps FastF1 marks as inaccurate, stints with too
# few clean laps) get STYLE_UNLABELED and are excluded from the style reward in
# f1_env.py and from style accuracy in evaluation.py.
STYLE_UNLABELED = -1
STYLE_PACE_THRESHOLD = 0.2  # seconds; gives roughly 25/50/25 % on 2023 data
MIN_CLEAN_LAPS_PER_STINT = 4


def compute_style_labels(laps, threshold=STYLE_PACE_THRESHOLD,
                         min_laps=MIN_CLEAN_LAPS_PER_STINT):
    """
    Return (style, residual) as numpy arrays aligned with the rows of `laps`
    (DataFrame with Track, Driver, Stint, LapNumber, LapTime [s], TyreLife,
    TrackStatus [decoded int], IsAccurate, Pit and optionally PitOutTime).
    style is 0/1/2 or STYLE_UNLABELED, residual is NaN where unlabeled.
    """
    import pandas as pd

    df = laps.reset_index(drop=True)
    clean = (
        df["IsAccurate"].astype(str).str.lower().isin(["true", "1", "1.0"])
        & (df["TrackStatus"].astype(int) == 1)
        & (df["Pit"].astype(int) == 0)
        & (df["LapNumber"] > 1)
        & df["LapTime"].notna()
    )
    if "PitOutTime" in df.columns:
        clean &= df["PitOutTime"].isna()

    style = np.full(len(df), STYLE_UNLABELED, dtype=np.int32)
    residual = np.full(len(df), np.nan)
    group_cols = [c for c in ("Year", "Track", "Driver", "Stint") if c in df.columns]
    for _, g in df[clean].groupby(group_cols):
        if len(g) < min_laps or g["TyreLife"].nunique() < 2:
            continue
        coef = np.polyfit(g["TyreLife"].astype(float), g["LapTime"].astype(float), 1)
        r = g["LapTime"].to_numpy(float) - np.polyval(coef, g["TyreLife"].to_numpy(float))
        residual[g.index] = r
        style[g.index] = np.where(r < -threshold, 2, np.where(r > threshold, 0, 1))
    return style, residual

# prepare_data.py
# Downloads lap and weather data with FastF1, builds the observation array and
# the ground-truth actions (pit, compound, style) and saves them to data/.
import os
import pandas as pd
import numpy as np
import fastf1
import matplotlib.pyplot as plt
from utils import (COMPOUND_TO_CODE, STYLE_UNLABELED, classify_weather,
                   compute_gaps, compute_style_labels, worst_track_status)

# Enable the FastF1 cache
os.makedirs("data/cache", exist_ok=True)
fastf1.Cache.enable_cache("data/cache")

# === Data from multiple races and drivers ===
# Events are given by their FULL official name. A short name such as 'Spa' is
# resolved by FastF1's fuzzy matching and silently picked the SPANISH Grand
# Prix (Barcelona, 66 laps) instead of the Belgian one; the check below makes
# such a mismatch fail loudly instead of producing wrongly labelled data.
races = [
    # (year, official event name, session, short track label used in data/)
    (2023, 'British Grand Prix', 'R', 'Silverstone'),
    (2023, 'Italian Grand Prix', 'R', 'Monza'),
    (2023, 'Belgian Grand Prix', 'R', 'Spa'),
]
drivers = ['VER', 'HAM', 'LEC', 'PER']
all_laps = []

for year, gp, session_type, track_label in races:
    session = fastf1.get_session(year, gp, session_type)
    if session.event["EventName"] != gp:
        raise ValueError(f"FastF1 resolved '{gp}' to '{session.event['EventName']}'")
    session.load()

    # Real weather data for the WHOLE session (shared by all drivers)
    weather_data = session.weather_data.copy().sort_values("Time").reset_index(drop=True)

    # REAL gaps to the car ahead / behind, computed from the timing-line
    # crossing times of ALL drivers (see compute_gaps in utils.py). This has to
    # happen before the laps are filtered to the four drivers below.
    session_laps = compute_gaps(pd.DataFrame(session.laps))

    for drv in drivers:
        laps = session_laps[session_laps["Driver"] == drv]
        laps = laps[laps['LapTime'].notnull() & laps['Time'].notnull()].copy()
        laps["Driver"] = drv
        laps["Year"] = year
        laps["Track"] = track_label
        laps = laps.sort_values("Time").reset_index(drop=True)

        # Join with the REAL weather data: merge_asof finds the nearest
        # weather_data sample in time for every lap
        laps = pd.merge_asof(
            laps, weather_data[["Time", "Rainfall", "TrackTemp", "AirTemp"]],
            on="Time", direction="nearest"
        )
        laps["TrackTemperature"] = laps["TrackTemp"]
        laps["AirTemperature"] = laps["AirTemp"]

        all_laps.append(laps)

laps = pd.concat(all_laps).reset_index(drop=True)

# Compound encoding uses the SAME convention as the agent's action space
# (0 SOFT, 1 MEDIUM, 2 HARD, 3 INTERMEDIATE, 4 WET), see COMPOUND_TO_CODE in utils.py
laps["CompoundEncoded"] = laps["Compound"].str.upper().map(COMPOUND_TO_CODE)
laps["CompoundEncoded"] = laps["CompoundEncoded"].fillna(1).astype(int)  # unknown -> MEDIUM

# Times in seconds
for col in ["LapTime", "Sector1Time", "Sector2Time"]:
    laps[col] = pd.to_timedelta(laps[col], errors='coerce').dt.total_seconds()

laps["Pit"] = laps["PitInTime"].notnull().astype(int)

# === REAL weather ===
# Uses the real Rainfall signal from FastF1 weather_data (joined above with
# merge_asof); see classify_weather() in utils.py for the meaning of codes 1/2/3.
laps["Weather"] = 1
for (drv, yr, trk), group in laps.groupby(["Driver", "Year", "Track"], sort=False):
    idx = group.index
    laps.loc[idx, "Weather"] = classify_weather(group["Rainfall"].tolist())

# CompoundChoice (ground truth for comparison with the driver) = the REAL tyre
# the driver used on that lap (same convention as CompoundEncoded above)
laps["CompoundChoice"] = laps["CompoundEncoded"]

# TrackStatus is stored by FastF1 as a STRING of concatenated codes; see
# worst_track_status() in utils.py for decoding
if "TrackStatus" in laps.columns:
    laps["TrackStatus"] = laps["TrackStatus"].apply(worst_track_status)

# Driving-style label from REAL pace relative to the driver's own stint trend
# (see compute_style_labels in utils.py). Needs the DECODED TrackStatus above.
# Laps without a meaningful label are STYLE_UNLABELED (-1) and are ignored by
# the style reward and by the style accuracy.
laps["Style"], laps["PaceResidual"] = compute_style_labels(laps)
n_labeled = int((laps["Style"] != STYLE_UNLABELED).sum())
print(f"Style labels: {n_labeled}/{len(laps)} laps labeled, distribution:",
      laps.loc[laps["Style"] != STYLE_UNLABELED, "Style"].value_counts().sort_index().to_dict())

# Features
selected_features = [
    "TyreLife", "CompoundEncoded", "LapNumber", "TrackStatus",
    "IsAccurate", "Position", "LapTime", "Sector1Time", "Sector2Time",
    "FrontGap", "RearGap", "Stint", "TrackTemperature", "AirTemperature", "Weather"
]

for col in selected_features:
    if col not in laps.columns:
        laps[col] = 0

laps[selected_features] = laps[selected_features].fillna(0)

# Save the data
laps.to_csv("data/f1_data.csv", index=False)
obs_array = laps[selected_features].to_numpy(dtype=np.float32)
true_actions = laps[["Pit", "CompoundChoice", "Style"]].to_numpy(dtype=np.int32)

np.save("data/obs_array.npy", obs_array)
np.save("data/true_actions.npy", true_actions)
# Track of every row, used for the leave-one-track-out evaluation
# (evaluate_holdout.py): train on two tracks, test on the third.
np.save("data/track_ids.npy", laps["Track"].to_numpy(dtype=str))

print("Data saved in .csv and .npy format.")

# Tyre strategy per driver and session
compound_colors = {'SOFT': 'red', 'MEDIUM': 'yellow', 'HARD': 'white',
                   'INTERMEDIATE': 'green', 'WET': 'blue'}

plt.figure(figsize=(12, 4))
for driver in drivers:
    driver_laps = laps[laps["Driver"] == driver]
    for stint, group in driver_laps.groupby("Stint"):
        start = group["LapNumber"].min()
        end = group["LapNumber"].max()
        compound = group["Compound"].iloc[0] if "Compound" in group else ""
        plt.barh(driver, width=end - start + 1, left=start,
                 color=compound_colors.get(compound.upper(), "gray"), edgecolor="black")

plt.title("Tyre strategy per lap (multiple drivers)")
plt.xlabel("Lap")
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()

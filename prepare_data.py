# prepare_data.py
# Downloads lap and weather data with FastF1, builds the observation array and
# the ground-truth actions (pit, compound, style) and saves them to data/.
import os
import pandas as pd
import numpy as np
import fastf1
import matplotlib.pyplot as plt
from utils import COMPOUND_TO_CODE, classify_weather, worst_track_status

# Enable the FastF1 cache
os.makedirs("data/cache", exist_ok=True)
fastf1.Cache.enable_cache("data/cache")

# === Data from multiple races and drivers ===
races = [
    (2023, 'Silverstone', 'R'),
    (2023, 'Monza', 'R'),
    (2023, 'Spa', 'R')
]
drivers = ['VER', 'HAM', 'LEC', 'PER']
all_laps = []

for year, gp, session_type in races:
    session = fastf1.get_session(year, gp, session_type)
    session.load()

    # Real weather data for the WHOLE session (shared by all drivers)
    weather_data = session.weather_data.copy().sort_values("Time").reset_index(drop=True)

    for drv in drivers:
        laps = session.laps.pick_driver(drv)
        laps = laps[laps['LapTime'].notnull()].copy()
        laps["Driver"] = drv
        laps["Year"] = year
        laps["Track"] = gp
        laps = laps.sort_values("Time").reset_index(drop=True)

        # Join with the REAL weather data: merge_asof finds the nearest
        # weather_data sample in time for every lap
        laps = pd.merge_asof(
            laps, weather_data[["Time", "Rainfall", "TrackTemp", "AirTemp"]],
            on="Time", direction="nearest"
        )
        # Use TrackTemp/AirTemp from weather_data if the lap data has none
        if "TrackTemperature" not in laps or laps["TrackTemperature"].isna().all():
            laps["TrackTemperature"] = laps["TrackTemp"]
        if "AirTemperature" not in laps or laps["AirTemperature"].isna().all():
            laps["AirTemperature"] = laps["AirTemp"]

        # FastF1 does not provide FrontGap/RearGap per lap, so they are synthetic
        if "FrontGap" not in laps:
            laps["FrontGap"] = np.random.uniform(0.5, 5.0, size=len(laps))
        if "RearGap" not in laps:
            laps["RearGap"] = np.random.uniform(0.5, 5.0, size=len(laps))

        # Add any missing key columns
        for col in ["FrontGap", "RearGap", "TrackTemperature", "AirTemperature"]:
             if col not in laps:
                 laps[col] = np.random.uniform(20.0, 35.0, size=len(laps)) if "Temperature" in col else np.random.uniform(0.5, 5.0, size=len(laps))

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

# Heuristic driving style (FastF1 has no direct "style" label)
def assign_style(row):
    if row["FrontGap"] < 1.5 or row["RearGap"] < 1.5:
        return 0  # conservative
    elif row["TyreLife"] < 8 or row["LapTime"] > 100:
        return 1  # neutral
    else:
        return 2  # aggressive

laps["Style"] = laps.apply(assign_style, axis=1)

# CompoundChoice (ground truth for comparison with the driver) = the REAL tyre
# the driver used on that lap (same convention as CompoundEncoded above)
laps["CompoundChoice"] = laps["CompoundEncoded"]

# TrackStatus is stored by FastF1 as a STRING of concatenated codes; see
# worst_track_status() in utils.py for decoding
if "TrackStatus" in laps.columns:
    laps["TrackStatus"] = laps["TrackStatus"].apply(worst_track_status)

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

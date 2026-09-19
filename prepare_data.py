# === prepare_data.py ===
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import fastf1
import matplotlib.pyplot as plt

# Postavi cache
os.makedirs("data/cache", exist_ok=True)
fastf1.Cache.enable_cache("data/cache")

# === Podaci iz više utrka i vozača ===
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
    for drv in drivers:
        laps = session.laps.pick_driver(drv)
        laps = laps[laps['LapTime'].notnull()].copy()
        laps["Driver"] = drv
        laps["Year"] = year
        laps["Track"] = gp
        # Provjera je li postoje FrontGap i RearGap
        if "FrontGap" not in laps:
            laps["FrontGap"] = np.random.uniform(0.5, 5.0, size=len(laps))
        if "RearGap" not in laps:
            laps["RearGap"] = np.random.uniform(0.5, 5.0, size=len(laps))
        
        # Provjera je li postoje ključne kolone, dodaj ako nedostaju
        for col in ["FrontGap", "RearGap", "TrackTemperature", "AirTemperature"]:
             if col not in laps:
                 laps[col] = np.random.uniform(20.0, 35.0, size=len(laps)) if "Temperature" in col else np.random.uniform(0.5, 5.0, size=len(laps))

            
        all_laps.append(laps)

laps = pd.concat(all_laps).reset_index(drop=True)

# Kodiranje compounda
compound_encoder = LabelEncoder()
laps["CompoundEncoded"] = compound_encoder.fit_transform(laps["Compound"])

# Vrijeme u sekunde
for col in ["LapTime", "Sector1Time", "Sector2Time"]:
    laps[col] = pd.to_timedelta(laps[col], errors='coerce').dt.total_seconds()

laps["Pit"] = laps["PitInTime"].notnull().astype(int)

# Generacija vremena
laps["Weather"] = np.random.choice([1, 2, 3], size=len(laps), p=[0.6, 0.3, 0.1])

# Heuristički stil vožnje
def assign_style(row):
    if row["FrontGap"] < 1.5 or row["RearGap"] < 1.5:
        return 0  # konzervativno
    elif row["TyreLife"] < 8 or row["LapTime"] > 100:
        return 1  # neutralno
    else:
        return 2  # agresivno

laps["Style"] = laps.apply(assign_style, axis=1)

# Heuristički compound izbor
def decide_compound(row):
    if row["Weather"] == 3:
        return 4  # WET
    elif row["Weather"] == 2:
        return 3  # INTER
    elif row["TrackTemperature"] > 32:
        return 0  # SOFT
    elif row["TrackTemperature"] < 20:
        return 2  # HARD
    else:
        return 1  # MEDIUM

laps["CompoundChoice"] = laps.apply(decide_compound, axis=1)

# Značajke
selected_features = [
    "TyreLife", "CompoundEncoded", "LapNumber", "TrackStatus",
    "IsAccurate", "Position", "LapTime", "Sector1Time", "Sector2Time",
    "FrontGap", "RearGap", "Stint", "TrackTemperature", "AirTemperature", "Weather"
]

for col in selected_features:
    if col not in laps.columns:
        laps[col] = 0

laps[selected_features] = laps[selected_features].fillna(0)

# Spremi podatke
laps.to_csv("data/f1_data.csv", index=False)
obs_array = laps[selected_features].to_numpy(dtype=np.float32)
true_actions = laps[["Pit", "CompoundChoice", "Style"]].to_numpy(dtype=np.int32)

np.save("data/obs_array.npy", obs_array)
np.save("data/true_actions.npy", true_actions)

print("Podaci spremljeni u .csv i .npy format.")

# Vizualizacija guma po vozačima i sesijama
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

plt.title("Strategija guma po krugu (više vozača)")
plt.xlabel("Krug")
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()

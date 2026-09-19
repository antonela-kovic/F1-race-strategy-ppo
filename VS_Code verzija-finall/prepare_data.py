# === prepare_data.py ===
import os
import pandas as pd
import numpy as np
import fastf1
import matplotlib.pyplot as plt
from utils import COMPOUND_TO_CODE, classify_weather, worst_track_status

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

    # Stvarni vremenski podaci za CIJELU sesiju (dijele ga svi vozači)
    weather_data = session.weather_data.copy().sort_values("Time").reset_index(drop=True)

    for drv in drivers:
        laps = session.laps.pick_driver(drv)
        laps = laps[laps['LapTime'].notnull()].copy()
        laps["Driver"] = drv
        laps["Year"] = year
        laps["Track"] = gp
        laps = laps.sort_values("Time").reset_index(drop=True)

        # Spoji sa STVARNIM vremenskim podacima (umjesto nasumičnog generiranja) -
        # merge_asof nalazi najbliži vremenski uzorak weather_data za svaki krug
        laps = pd.merge_asof(
            laps, weather_data[["Time", "Rainfall", "TrackTemp", "AirTemp"]],
            on="Time", direction="nearest"
        )
        # Preferiraj stvarne TrackTemp/AirTemp iz weather_data ako lap podaci nemaju svoje
        if "TrackTemperature" not in laps or laps["TrackTemperature"].isna().all():
            laps["TrackTemperature"] = laps["TrackTemp"]
        if "AirTemperature" not in laps or laps["AirTemperature"].isna().all():
            laps["AirTemperature"] = laps["AirTemp"]

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

# Kodiranje compounda - koristi ISTU konvenciju kao akcijski prostor agenta
# (0 SOFT, 1 MEDIUM, 2 HARD, 3 INTERMEDIATE, 4 WET), a ne proizvoljni
# alfabetski poredak LabelEncoder-a (koji bi HARD stavio na 0, itd.)
# (COMPOUND_TO_CODE se sad uvozi iz utils.py, dijeli ga i logic_layer.py)
laps["CompoundEncoded"] = laps["Compound"].str.upper().map(COMPOUND_TO_CODE)
laps["CompoundEncoded"] = laps["CompoundEncoded"].fillna(1).astype(int)  # nepoznato -> MEDIUM

# Vrijeme u sekunde
for col in ["LapTime", "Sector1Time", "Sector2Time"]:
    laps[col] = pd.to_timedelta(laps[col], errors='coerce').dt.total_seconds()

laps["Pit"] = laps["PitInTime"].notnull().astype(int)

# === STVARNO vrijeme (umjesto nasumičnog generiranja) ===
# Koristimo pravi Rainfall signal iz FastF1 weather_data (spojen ranije preko merge_asof)
# - vidi classify_weather() u utils.py za značenje kodova 1/2/3.
laps["Weather"] = 1
for (drv, yr, trk), group in laps.groupby(["Driver", "Year", "Track"], sort=False):
    idx = group.index
    laps.loc[idx, "Weather"] = classify_weather(group["Rainfall"].tolist())

# Heuristički stil vožnje
def assign_style(row):
    if row["FrontGap"] < 1.5 or row["RearGap"] < 1.5:
        return 0  # konzervativno
    elif row["TyreLife"] < 8 or row["LapTime"] > 100:
        return 1  # neutralno
    else:
        return 2  # agresivno

laps["Style"] = laps.apply(assign_style, axis=1)

# CompoundChoice (ground truth za usporedbu s vozačem) = STVARNA guma koju
# je vozač koristio u tom krugu (ista konvencija kao CompoundEncoded gore),
# umjesto ranije heurističke funkcije koja je ovisila o nasumičnom Weatheru.
laps["CompoundChoice"] = laps["CompoundEncoded"]

# === TrackStatus: FastF1 ga sprema kao STRING SPOJENIH kodova - vidi
# worst_track_status() u utils.py za objašnjenje i dekodiranje.
if "TrackStatus" in laps.columns:
    laps["TrackStatus"] = laps["TrackStatus"].apply(worst_track_status)

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

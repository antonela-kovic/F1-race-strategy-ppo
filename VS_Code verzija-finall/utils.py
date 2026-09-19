# === utils.py ===
import numpy as np

# Može se ukloniti ova funkcija(generate_weather_sequence) jer je trenutno train.py prešao na Weather 
# kao značajku u prepare_data.py te se više ne koristi u f1_env.py.

# def generate_weather_sequence(num_laps):
#     weather = [1]
#     for _ in range(1, num_laps):
#         last = weather[-1]
#         change = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
#         new = min(max(last + change, 1), 3)
#         weather.append(new)
#     return weather

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


# === Skaliranje opažanja (normalizacija) ===
# Značajke su vrlo različitih raspona (npr. LapTime~80-150 vs FrontGap 0.5-5),
# a mreža bez normalizacije efektivno "ne vidi" značajke s malim rasponom u
# odnosu na one s velikim - to je bio uzrok zašto je RAW politika naučila
# razlikovati compound (ovisi o TyreLife/TrackTemperature, veći raspon), ali
# NE i stil vožnje (ovisi gotovo isključivo o FrontGap/RearGap, mali raspon
# ~0.5-5 nasuprot npr. LapNumber-u raspona ~1-70).
#
# KORISTI OVU ISTU funkciju posvuda gdje se opažanje šalje modelu -
# f1_env.py (trening), predict_strategies.py i api.py (inferencija) - jer bi
# model treniran na skaliranim opažanjima davao besmislene predikcije na
# neskaliranim i obrnuto. verify_action() u logic_layer.py i dalje prima
# SIROVO (neskalirano) stanje, jer njegovi pragovi (npr. tyre_life < 2.0,
# front_gap < 1.5) pretpostavljaju stvarne jedinice.
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
    Vrati SKALIRANU verziju opažanja (dijeli svaku značajku njenom skalom iz
    FEATURE_SCALE, po istom redoslijedu kao selected_features), tako da sve
    značajke budu otprilike u sličnom rasponu (~0-1.5). Radi i na jednom
    retku (shape (15,)) i na 2D nizu redaka (shape (N,15)).
    """
    arr = np.asarray(obs, dtype=np.float32)
    return arr / _SCALE_VECTOR


# === Čiste pomoćne funkcije za pripremu podataka (prepare_data.py) ===
# Premještene ovamo (umjesto da žive u prepare_data.py) jer prepare_data.py
# na vrhu datoteke odmah pokreće fastf1.get_session(...).load() - modul se
# NE MOŽE sigurno importati (ni radi testiranja ni iz drugog koda) bez da to
# odmah pokrene mrežne pozive. Ove funkcije su čisti Python (bez pandas/
# fastf1 ovisnosti) pa se mogu jedinično testirati izolirano.

# Ista konvencija kao akcijski prostor agenta (0 SOFT, 1 MEDIUM, 2 HARD,
# 3 INTERMEDIATE, 4 WET), a ne proizvoljni alfabetski poredak LabelEncoder-a.
COMPOUND_TO_CODE = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}


def classify_weather(rainfall_series, damp_memory_laps=3):
    """
    Pretvori niz bool/truthy Rainfall vrijednosti (jedna po krugu, kronološkim
    redoslijedom unutar JEDNE utrke/vozača) u niz Weather kodova:
      3 = kiša (Rainfall aktivan taj krug)
      2 = vlažno/damp (kiša je nedavno prestala - staza je još mokra, traje
          `damp_memory_laps` krugova nakon zadnjeg kruga s kišom)
      1 = suho (nikad nije padala kiša, ili je davno prestala)
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


# FastF1 TrackStatus dolazi kao STRING SPOJENIH kodova koji su se dogodili
# tijekom kruga (npr. "41" znači da su se dogodila i status 4 I status 1,
# "1264" znači 1,2,6,4 - NE brojevi 41 ili 1264!). Prioritet za dekodiranje
# u JEDAN najozbiljniji status: 5=crveni najozbiljniji, zatim SC, VSC,
# VSC kraj, žuti, čisto.
STATUS_PRIORITY = [5, 4, 6, 7, 2, 1]


def worst_track_status(raw) -> int:
    """Dekodiraj FastF1-ov spojeni TrackStatus string u jedan (najozbiljniji)
    pojedinačni status kod. Vraća 1 (čisto) za nepoznat/prazan ulaz."""
    digits = set(str(raw))
    for code in STATUS_PRIORITY:
        if str(code) in digits:
            return code
    return 1

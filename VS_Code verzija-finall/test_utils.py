# === test_utils.py ===
# Jedinični testovi za čiste pomoćne funkcije u utils.py. Pokreni s:
#   pytest test_utils.py -v
import numpy as np
import pytest

from utils import (
    classify_weather,
    worst_track_status,
    scale_observation,
    selected_features,
    COMPOUND_TO_CODE,
    STATUS_PRIORITY,
)


# --- classify_weather ---

def test_classify_weather_all_dry():
    assert classify_weather([False, False, False, False]) == [1, 1, 1, 1]


def test_classify_weather_rain_then_damp_then_dry():
    # kiša 2 kruga, pa 3 kruga "pamćenja" vlage (damp_memory_laps=3 default),
    # pa natrag na suho
    rainfall = [False, True, True, False, False, False, False]
    result = classify_weather(rainfall, damp_memory_laps=3)
    assert result == [1, 3, 3, 2, 2, 2, 1]


def test_classify_weather_damp_memory_is_configurable():
    rainfall = [True, False, False, False]
    assert classify_weather(rainfall, damp_memory_laps=1) == [3, 2, 1, 1]
    assert classify_weather(rainfall, damp_memory_laps=0) == [3, 1, 1, 1]


def test_classify_weather_empty_input():
    assert classify_weather([]) == []


# --- worst_track_status ---

@pytest.mark.parametrize("raw,expected", [
    ("1", 1),      # čisto
    ("2", 2),      # žuti
    ("4", 4),      # Safety Car
    ("41", 4),     # SC ima prednost nad "čisto"
    ("126", 6),    # VSC(6) ima prednost nad žutim(2) i čistim(1)
    ("1264", 4),   # Safety Car ima prednost nad svim ostalim prisutnim
    ("", 1),       # prazan string -> tretiraj kao čisto
    ("9", 1),      # nepoznat kod -> tretiraj kao čisto
])
def test_worst_track_status(raw, expected):
    assert worst_track_status(raw) == expected


def test_worst_track_status_priority_order_is_red_first():
    # Crveni (5) mora imati prednost nad svima, uključujući Safety Car (4)
    assert worst_track_status("45") == 5
    assert STATUS_PRIORITY[0] == 5


# --- COMPOUND_TO_CODE ---

def test_compound_to_code_matches_action_space_convention():
    # Ova konvencija MORA se poklapati s onom u logic_layer.py
    # (COMPOUND_NAMES) i s akcijskim prostorom agenta (f1_env.py)
    assert COMPOUND_TO_CODE == {
        "SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4
    }


# --- scale_observation ---

def test_scale_observation_shape_preserved_single_row():
    row = np.zeros(len(selected_features))
    scaled = scale_observation(row)
    assert scaled.shape == row.shape


def test_scale_observation_shape_preserved_2d():
    arr = np.zeros((10, len(selected_features)))
    scaled = scale_observation(arr)
    assert scaled.shape == arr.shape


def test_scale_observation_known_values_land_in_reasonable_range():
    # Realni raspon vrijednosti (vidi FEATURE_SCALE u utils.py) - nakon
    # skaliranja sve bi trebalo biti otprilike u [0, 1.6], ne u stotinama
    # kao sirove vrijednosti (npr. LapTime~90-150).
    row = np.array([
        20,    # TyreLife
        1,     # CompoundEncoded
        30,    # LapNumber
        1,     # TrackStatus
        1,     # IsAccurate
        5,     # Position
        95.0,  # LapTime
        30.0,  # Sector1Time
        32.0,  # Sector2Time
        2.0,   # FrontGap
        2.0,   # RearGap
        1,     # Stint
        35.0,  # TrackTemperature
        25.0,  # AirTemperature
        1,     # Weather
    ])
    scaled = scale_observation(row)
    assert (scaled >= 0).all()
    assert (scaled <= 1.6).all()


def test_scale_observation_is_deterministic():
    row = np.arange(len(selected_features), dtype=np.float32)
    assert np.allclose(scale_observation(row), scale_observation(row))

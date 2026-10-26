# test_utils.py
# Unit tests for the pure helper functions in utils.py. Run with:
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
    # rain for 2 laps, then 3 laps of damp "memory" (damp_memory_laps=3 default),
    # then back to dry
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
    ("1", 1),      # clear
    ("2", 2),      # yellow
    ("4", 4),      # Safety Car
    ("41", 4),     # SC takes priority over "clear"
    ("126", 6),    # VSC(6) takes priority over yellow(2) and clear(1)
    ("1264", 4),   # Safety Car takes priority over all other present codes
    ("", 1),       # empty string -> treat as clear
    ("9", 1),      # unknown code -> treat as clear
])
def test_worst_track_status(raw, expected):
    assert worst_track_status(raw) == expected


def test_worst_track_status_priority_order_is_red_first():
    # Red (5) must take priority over everything, including Safety Car (4)
    assert worst_track_status("45") == 5
    assert STATUS_PRIORITY[0] == 5


# --- COMPOUND_TO_CODE ---

def test_compound_to_code_matches_action_space_convention():
    # This convention MUST match safety_layer.py (COMPOUND_NAMES)
    # and the agent's action space (f1_env.py)
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
    # Realistic value ranges (see FEATURE_SCALE in utils.py): after scaling,
    # everything should be roughly in [0, 1.6], not in the hundreds like the
    # raw values (e.g. LapTime ~90-150).
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

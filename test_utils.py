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


# --- compute_gaps (real FrontGap / RearGap from timing-line crossings) ---

def _laps_frame(rows):
    import pandas as pd
    return pd.DataFrame(rows, columns=["Driver", "LapNumber", "Position", "Time"])


def test_compute_gaps_uses_crossing_time_of_neighbours():
    from utils import compute_gaps
    laps = _laps_frame([
        ("A", 1, 1, 100.0), ("B", 1, 2, 101.5), ("C", 1, 3, 104.0),
        ("A", 2, 1, 190.0), ("C", 2, 2, 190.8), ("B", 2, 3, 191.0),
    ])
    out = compute_gaps(laps, cap=10.0).set_index(["Driver", "LapNumber"])
    assert out.loc[("B", 1), "FrontGap"] == pytest.approx(1.5)
    assert out.loc[("B", 1), "RearGap"] == pytest.approx(2.5)
    assert out.loc[("C", 2), "FrontGap"] == pytest.approx(0.8)   # overtook B
    assert out.loc[("B", 2), "FrontGap"] == pytest.approx(0.2)


def test_compute_gaps_leader_last_and_missing_get_cap():
    from utils import compute_gaps
    laps = _laps_frame([
        ("A", 1, 1, 100.0), ("B", 1, 2, 130.0),   # 30 s gap -> clipped
        ("C", 1, 4, 140.0),                        # P3 missing -> no neighbour
        ("D", 1, None, 150.0),                     # no position
    ])
    out = compute_gaps(laps, cap=10.0).set_index("Driver")
    assert out.loc["A", "FrontGap"] == 10.0
    assert out.loc["B", "FrontGap"] == 10.0
    assert out.loc["C", "FrontGap"] == 10.0 and out.loc["C", "RearGap"] == 10.0
    assert out.loc["D", "FrontGap"] == 10.0


def test_compute_gaps_accepts_timedelta():
    import pandas as pd
    from utils import compute_gaps
    laps = _laps_frame([("A", 1, 1, 100.0), ("B", 1, 2, 102.0)])
    laps["Time"] = pd.to_timedelta(laps["Time"], unit="s")
    out = compute_gaps(laps)
    assert out["FrontGap"].tolist() == pytest.approx([10.0, 2.0])


# --- compute_style_labels (pace relative to the stint trend) ---

def _stint_frame(lap_times, track_status=None, pit=None, accurate=None):
    import pandas as pd
    n = len(lap_times)
    return pd.DataFrame({
        "Track": "X", "Driver": "D", "Stint": 1,
        "LapNumber": np.arange(2, n + 2),
        "TyreLife": np.arange(1, n + 1, dtype=float),
        "LapTime": lap_times,
        "TrackStatus": track_status or [1] * n,
        "Pit": pit or [0] * n,
        "IsAccurate": accurate or [True] * n,
    })


def test_style_labels_follow_residual_sign():
    from utils import compute_style_labels
    # linear trend 90 + 0.1*lap, lap 4 is 1 s faster, lap 6 is 1 s slower
    times = [90.1, 90.2, 90.3, 89.4, 90.5, 91.6, 90.7, 90.8]
    style, residual = compute_style_labels(_stint_frame(times))
    assert style[3] == 2          # pushing -> aggressive
    assert style[5] == 0          # slow -> conservative
    assert (style[[0, 1, 2, 4, 6, 7]] == 1).sum() >= 4
    assert not np.isnan(residual).any()


def test_style_labels_skip_caution_pit_and_inaccurate_laps():
    from utils import compute_style_labels, STYLE_UNLABELED
    times = [90.0] * 8
    df = _stint_frame(times, track_status=[1, 4, 1, 1, 1, 1, 1, 1],
                      pit=[0, 0, 1, 0, 0, 0, 0, 0],
                      accurate=[True, True, True, False, True, True, True, True])
    style, _ = compute_style_labels(df)
    assert style[1] == STYLE_UNLABELED  # Safety Car
    assert style[2] == STYLE_UNLABELED  # pit lap
    assert style[3] == STYLE_UNLABELED  # inaccurate
    assert (style[[0, 4, 5, 6, 7]] == 1).all()


def test_style_labels_short_stint_is_unlabeled():
    from utils import compute_style_labels, STYLE_UNLABELED
    style, residual = compute_style_labels(_stint_frame([90.0, 90.1, 90.2]))
    assert (style == STYLE_UNLABELED).all()
    assert np.isnan(residual).all()

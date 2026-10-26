# test_safety_layer.py
# Unit tests for safety_layer.py (rules R1-R10 and RaceState). Run with:
#   pytest test_safety_layer.py -v
import numpy as np
import pytest

from safety_layer import verify_action, RaceState, humanize_action

SELECTED_FEATURES = [
    "TyreLife", "CompoundEncoded", "LapNumber", "TrackStatus",
    "IsAccurate", "Position", "LapTime", "Sector1Time", "Sector2Time",
    "FrontGap", "RearGap", "Stint", "TrackTemperature", "AirTemperature",
    "Weather",
]


def make_state(**overrides):
    """Build a state vector with sensible defaults (dry, good tyres, normal
    gap), then override only what the test needs."""
    defaults = {
        "TyreLife": 15, "CompoundEncoded": 1, "LapNumber": 20, "TrackStatus": 1,
        "IsAccurate": 1, "Position": 5, "LapTime": 95.0, "Sector1Time": 30.0,
        "Sector2Time": 32.0, "FrontGap": 3.0, "RearGap": 3.0, "Stint": 1,
        "TrackTemperature": 25.0, "AirTemperature": 20.0, "Weather": 1,
    }
    defaults.update(overrides)
    return np.array([defaults[f] for f in SELECTED_FEATURES], dtype=np.float32)


# --- R1: rain -> WET ---

def test_r1_wet_weather_forces_wet_tyres():
    state = make_state(Weather=3)
    action = np.array([0, 0, 0])  # SOFT in the rain - violation
    safe, report = verify_action(state, action, SELECTED_FEATURES)
    assert safe[1] == 4  # WET
    assert any(o["rule"] == "R1_WET_REQUIRES_WET" for o in report.overrides)


def test_r1_no_override_if_already_wet():
    state = make_state(Weather=3)
    action = np.array([0, 4, 0])  # already WET
    safe, report = verify_action(state, action, SELECTED_FEATURES)
    assert safe[1] == 4
    assert not any(o["rule"] == "R1_WET_REQUIRES_WET" for o in report.overrides)


# --- R2: damp -> INTER ---

def test_r2_damp_weather_forces_inter():
    state = make_state(Weather=2)
    action = np.array([0, 1, 0])  # MEDIUM on a damp track - violation
    safe, report = verify_action(state, action, SELECTED_FEATURES)
    assert safe[1] == 3  # INTER


# --- R3a: dry forbids INTER/WET ---

@pytest.mark.parametrize("bad_compound", [3, 4])
def test_r3a_dry_forbids_wet_compounds(bad_compound):
    state = make_state(Weather=1)
    action = np.array([0, bad_compound, 0])
    safe, report = verify_action(state, action, SELECTED_FEATURES)
    assert safe[1] not in (3, 4)
    assert any(o["rule"] == "R3a_DRY_FORBIDS_INTER_WET" for o in report.overrides)


def test_r3a_allows_all_dry_compounds_unchanged():
    for comp in (0, 1, 2):
        state = make_state(Weather=1)
        action = np.array([0, comp, 0])
        safe, _ = verify_action(state, action, SELECTED_FEATURES)
        assert safe[1] == comp  # HARD is not rewritten on a dry track


# --- R4: critically low tyre life forces a pit ---

def test_r4_forces_pit_when_tyres_critical():
    state = make_state(TyreLife=1.0)
    action = np.array([0, 1, 0])
    safe, report = verify_action(state, action, SELECTED_FEATURES)
    assert safe[0] == 1
    assert any(o["rule"] == "R4_LOW_TYRE_FORCES_PIT" for o in report.overrides)


# --- R5: block a pointless pit on a dry track with good tyres ---

def test_r5_blocks_pointless_pit():
    state = make_state(Weather=1, TyreLife=35.0)
    action = np.array([1, 1, 0])
    safe, report = verify_action(state, action, SELECTED_FEATURES)
    assert safe[0] == 0
    assert any(o["rule"] == "R5_BLOCK_POINTLESS_PIT" for o in report.overrides)


# --- R6: wet/damp forbids an aggressive style ---

@pytest.mark.parametrize("weather", [2, 3])
def test_r6_wet_or_damp_forbids_aggressive(weather):
    state = make_state(Weather=weather)
    action = np.array([0, 1, 2])
    safe, report = verify_action(state, action, SELECTED_FEATURES)
    assert safe[2] != 2
    assert any(o["rule"] == "R6_WET_AVOIDS_AGGRESSIVE" for o in report.overrides)


# --- R7: small gap -> conservative ---

def test_r7_small_gap_forces_conservative():
    state = make_state(FrontGap=0.5)
    action = np.array([0, 1, 2])
    safe, report = verify_action(state, action, SELECTED_FEATURES)
    assert safe[2] == 0
    assert any(o["rule"] == "R7_SMALL_GAP_CONSERVATIVE" for o in report.overrides)


# --- R9: Safety Car/VSC/red/yellow forbid an aggressive style ---

@pytest.mark.parametrize("track_status", [2, 4, 5, 6, 7])
def test_r9_caution_forbids_aggressive(track_status):
    state = make_state(TrackStatus=track_status, FrontGap=3.0)  # gap large enough that R7 does not interfere
    action = np.array([0, 1, 2])
    safe, report = verify_action(state, action, SELECTED_FEATURES)
    assert safe[2] == 0
    assert any(o["rule"] == "R9_CAUTION_FORBIDS_AGGRESSIVE" for o in report.overrides)


def test_r9_clear_track_allows_aggressive():
    state = make_state(TrackStatus=1, FrontGap=3.0)
    action = np.array([0, 1, 2])
    safe, report = verify_action(state, action, SELECTED_FEATURES)
    assert safe[2] == 2
    assert not any(o["rule"] == "R9_CAUTION_FORBIDS_AGGRESSIVE" for o in report.overrides)


# --- R10: two mandatory dry compounds in a dry race ---

def test_r10_forces_pit_and_compound_change_near_race_end():
    race_state = RaceState(total_laps=50, force_window=3)
    for _ in range(47):
        race_state.record(weather=1, compound=1)  # MEDIUM only for the whole race
    assert race_state.laps_remaining == 2

    state = make_state(Weather=1, TyreLife=20.0)  # tyres OK, the agent would normally not pit
    action = np.array([0, 1, 0])  # stays on the same compound, no pit
    safe, report = verify_action(state, action, SELECTED_FEATURES, race_state=race_state)

    assert safe[0] == 1  # pit forced
    assert safe[1] != 1  # compound change forced
    assert any(o["rule"] == "R10_MANDATORY_TWO_DRY_COMPOUNDS" for o in report.overrides)


def test_r10_not_triggered_if_race_was_ever_wet():
    race_state = RaceState(total_laps=50, force_window=3)
    for _ in range(20):
        race_state.record(weather=1, compound=1)
    race_state.record(weather=3, compound=4)  # rain somewhere in the middle of the race
    for _ in range(26):
        race_state.record(weather=1, compound=1)

    state = make_state(Weather=1, TyreLife=20.0)
    action = np.array([0, 1, 0])
    safe, report = verify_action(state, action, SELECTED_FEATURES, race_state=race_state)

    assert safe[0] == 0  # no forced pit - the race counts as a "wet race"
    assert not any(o["rule"] == "R10_MANDATORY_TWO_DRY_COMPOUNDS" for o in report.overrides)


def test_r10_not_triggered_when_two_compounds_already_used():
    race_state = RaceState(total_laps=50, force_window=3)
    for _ in range(20):
        race_state.record(weather=1, compound=0)
    for _ in range(25):
        race_state.record(weather=1, compound=1)  # already two different compounds

    state = make_state(Weather=1, TyreLife=20.0)
    action = np.array([0, 1, 0])
    safe, report = verify_action(state, action, SELECTED_FEATURES, race_state=race_state)

    assert not any(o["rule"] == "R10_MANDATORY_TWO_DRY_COMPOUNDS" for o in report.overrides)


def test_r10_skipped_entirely_without_race_state():
    # Without race_state (e.g. a single API call without context), R10 is
    # skipped instead of failing.
    state = make_state(Weather=1, TyreLife=20.0)
    action = np.array([0, 1, 0])
    safe, report = verify_action(state, action, SELECTED_FEATURES, race_state=None)
    assert not any(o["rule"] == "R10_MANDATORY_TWO_DRY_COMPOUNDS" for o in report.overrides)


# --- RaceState properties ---

def test_race_state_laps_remaining():
    rs = RaceState(total_laps=10)
    assert rs.laps_remaining == 9
    rs.record(weather=1, compound=0)
    assert rs.laps_remaining == 8


def test_race_state_dry_compounds_used_excludes_wet_inter():
    rs = RaceState(total_laps=10)
    rs.record(weather=1, compound=0)
    rs.record(weather=3, compound=4)  # WET does not count as a dry compound
    assert rs.dry_compounds_used == {0}


def test_race_state_is_dry_race_so_far():
    rs = RaceState(total_laps=10)
    assert rs.is_dry_race_so_far is True
    rs.record(weather=2, compound=0)
    assert rs.is_dry_race_so_far is False


# --- humanize_action ---

def test_humanize_action_readable_output():
    result = humanize_action(np.array([1, 4, 2]))
    assert result["pit"] == "YES"
    assert "WET" in result["compound"]
    assert "Aggressive" in result["style"]

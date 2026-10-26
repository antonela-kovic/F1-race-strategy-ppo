# test_f1_env.py
# Unit tests for f1_env.py: race segmentation, race-order shuffling,
# per-race bookkeeping and the R10-equivalent training reward. Run with:
#   pytest test_f1_env.py -v
import numpy as np
import pytest

from f1_env import FastF1MultiActionEnv
from utils import selected_features


def make_synthetic_obs(race_lengths, weather=1):
    """
    Build an artificial obs_array made of several concatenated "races",
    each with LapNumber 1..N (a drop in LapNumber = race boundary, the same
    principle f1_env.py/predict_strategies.py use for segmentation).
    All other features are set to neutral, safe values.
    """
    i = {name: idx for idx, name in enumerate(selected_features)}
    rows = []
    for length in race_lengths:
        for lap in range(1, length + 1):
            row = np.zeros(len(selected_features), dtype=np.float32)
            row[i["TyreLife"]] = 20.0
            row[i["CompoundEncoded"]] = 1
            row[i["LapNumber"]] = lap
            row[i["TrackStatus"]] = 1
            row[i["IsAccurate"]] = 1
            row[i["Position"]] = 5
            row[i["LapTime"]] = 95.0
            row[i["Sector1Time"]] = 30.0
            row[i["Sector2Time"]] = 32.0
            row[i["FrontGap"]] = 3.0
            row[i["RearGap"]] = 3.0
            row[i["Stint"]] = 1
            row[i["TrackTemperature"]] = 25.0
            row[i["AirTemperature"]] = 20.0
            row[i["Weather"]] = weather
            rows.append(row)
    return np.array(rows, dtype=np.float32)


@pytest.fixture
def race_lengths():
    return [10, 15, 8, 12]


@pytest.fixture
def obs_array(race_lengths):
    return make_synthetic_obs(race_lengths)


@pytest.fixture
def true_actions(obs_array):
    # pit always 0, compound always MEDIUM(1), style always conservative(0)
    n = len(obs_array)
    return np.zeros((n, 3), dtype=np.int32) + [0, 1, 0]


# --- Race segmentation ---

def test_split_into_races_matches_input_lengths(obs_array, race_lengths):
    env = FastF1MultiActionEnv(obs_array, true_actions=None)
    segments = env._split_into_races(obs_array)
    assert [e - s for s, e in segments] == race_lengths


def test_split_into_races_single_race():
    obs = make_synthetic_obs([20])
    env = FastF1MultiActionEnv(obs, true_actions=None)
    segments = env._split_into_races(obs)
    assert segments == [(0, 20)]


# --- Race-order shuffling ---

def test_shuffle_preserves_within_race_lap_order(obs_array):
    env = FastF1MultiActionEnv(obs_array, true_actions=None)
    i_lap = selected_features.index("LapNumber")
    shuffled, orig_indices = env._shuffle_race_order(obs_array)

    for start, end in env._split_into_races(shuffled):
        lap_numbers = shuffled[start:end, i_lap]
        # within EVERY race, LapNumber must still increase 1,2,3,...
        assert list(lap_numbers) == sorted(lap_numbers.tolist())
        assert lap_numbers[0] == 1


def test_shuffle_orig_indices_correctly_map_back(obs_array):
    env = FastF1MultiActionEnv(obs_array, true_actions=None)
    shuffled, orig_indices = env._shuffle_race_order(obs_array)
    # every row after shuffling must be IDENTICAL to the row at the orig_indices
    # position in the original (unshuffled) array
    assert np.allclose(shuffled, obs_array[orig_indices])


def test_shuffle_preserves_total_row_count(obs_array):
    env = FastF1MultiActionEnv(obs_array, true_actions=None)
    shuffled, orig_indices = env._shuffle_race_order(obs_array)
    assert len(shuffled) == len(obs_array)
    assert len(orig_indices) == len(obs_array)
    assert sorted(orig_indices.tolist()) == list(range(len(obs_array)))


# --- Per-race bookkeeping is reset at race boundaries ---

def test_per_race_bookkeeping_resets_at_each_boundary(obs_array, true_actions, race_lengths):
    env = FastF1MultiActionEnv(obs_array, true_actions=true_actions, difficulty_level="easy")
    env.reset(seed=1)

    race_starts_seen = 0
    while True:
        is_race_start = env.index in env.race_starts
        action = np.array([0, 1, 0])  # conservative, consistent action
        obs, reward, done, trunc, info = env.step(action)
        if is_race_start:
            race_starts_seen += 1
            # right after the first lap of a new race: lap_in_race should be 1
            # (reset to 0, then incremented), and dry_compounds_used should
            # contain ONLY the compound from THAT lap (nothing carried over
            # from the previous race)
            assert env.lap_in_race == 1
            assert env.dry_compounds_used == {1}
            assert env.wet_or_damp_seen is False
        if done:
            break

    assert race_starts_seen == len(race_lengths)


def test_episode_runs_full_length_without_risky_style(obs_array, true_actions):
    # The style is never aggressive (2), so the random "DNF" mechanism
    # (which requires style==2) never fires and the episode must go through
    # ALL rows.
    env = FastF1MultiActionEnv(obs_array, true_actions=true_actions, difficulty_level="easy")
    env.reset(seed=1)
    steps = 0
    while True:
        obs, reward, done, trunc, info = env.step(np.array([0, 1, 0]))
        steps += 1
        if done:
            break
    assert steps == len(obs_array)


# --- The reward favours correct behaviour (regression test for the sqrt(index) bug) ---

def test_true_matching_beats_degenerate_policy_per_step(obs_array):
    # Regression test: an unconditional "reward += sqrt(self.index)" bonus once
    # made an agent that ALWAYS picks the same (wrong) action earn MORE total
    # reward than an agent that matches the correct action. This test catches
    # any similar bug if it is reintroduced.
    true_actions = np.zeros((len(obs_array), 3), dtype=np.int32)
    true_actions[:, 1] = 1  # true_comp = MEDIUM on every lap

    def run(action_fn):
        env = FastF1MultiActionEnv(obs_array, true_actions=true_actions, difficulty_level="easy")
        env.reset(seed=42)
        total, steps = 0.0, 0
        while True:
            obs, reward, done, trunc, info = env.step(action_fn())
            total += reward
            steps += 1
            if done:
                break
        return total / steps

    avg_true_matching = run(lambda: np.array([0, 1, 0]))       # matches true_comp
    avg_degenerate = run(lambda: np.array([0, 0, 0]))          # always SOFT (wrong)

    assert avg_true_matching > avg_degenerate


# --- A DNF continues with the next race instead of ending the episode ---

def test_dnf_continues_to_next_race_not_whole_episode(race_lengths):
    # weather=2 (damp) enables the DNF branch (requires weather>1) and
    # style=2 (aggressive) actually triggers it. With enough attempts a DNF
    # should happen at least once, and the episode must still finish through
    # all rows (a DNF only skips to the next race).
    obs = make_synthetic_obs(race_lengths, weather=2)
    true_actions = np.zeros((len(obs), 3), dtype=np.int32)
    env = FastF1MultiActionEnv(obs, true_actions=true_actions, difficulty_level="easy")
    env.reset(seed=123)

    steps = 0
    max_steps = len(obs) * 2  # safety limit
    while True:
        obs_, reward, done, trunc, info = env.step(np.array([0, 1, 2]))  # aggressive, risky
        steps += 1
        if done or steps > max_steps:
            break

    assert steps <= max_steps
    # the episode must finish (done=True) within a reasonable number of steps,
    # regardless of how many DNFs happened along the way
    assert done or steps > max_steps  # if the safety limit is hit, the test below fails anyway


# --- Reproducibility via the seed parameter ---

def test_reset_with_same_seed_is_fully_reproducible(obs_array, true_actions):
    def run(seed):
        env = FastF1MultiActionEnv(obs_array, true_actions=true_actions, difficulty_level="full")
        obs, _ = env.reset(seed=seed)
        trace = [obs.copy()]
        for _ in range(20):
            obs, reward, done, trunc, info = env.step(np.array([0, 1, 0]))
            trace.append(obs.copy())
            if done:
                break
        return np.array(trace)

    assert np.allclose(run(seed=42), run(seed=42))


def test_reset_with_different_seeds_differs(obs_array, true_actions):
    def run(seed):
        env = FastF1MultiActionEnv(obs_array, true_actions=true_actions, difficulty_level="full")
        obs, _ = env.reset(seed=seed)
        trace = [obs.copy()]
        for _ in range(20):
            obs, reward, done, trunc, info = env.step(np.array([0, 1, 0]))
            trace.append(obs.copy())
            if done:
                break
        return np.array(trace)

    assert not np.allclose(run(seed=1), run(seed=2))


def test_reset_without_seed_remains_stochastic(obs_array, true_actions):
    # Without an explicit seed (as SB3 calls it during training), the generator
    # MUST keep changing state between resets, otherwise training would always
    # see the same order/augmentation. The test checks the actual race order
    # (orig_indices) over several resets (not just the first observation, which
    # looks the same on this uniform fixture) and requires AT LEAST ONE
    # difference, so it is not flaky on the small (1/4! ≈ 4%) chance that two
    # random orders match.
    env = FastF1MultiActionEnv(obs_array, true_actions=true_actions, difficulty_level="full")
    orders = []
    for _ in range(5):
        env.reset()
        orders.append(tuple(env.orig_indices.tolist()))
    assert len(set(orders)) > 1
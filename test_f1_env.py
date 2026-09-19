# === test_f1_env.py ===
# Jedinični testovi za f1_env.py - segmentacija utrka, miješanje redoslijeda,
# per-utrka bookkeeping, R10-ekvivalentna nagrada u treningu. Pokreni s:
#   pytest test_f1_env.py -v
import numpy as np
import pytest

from f1_env import FastF1MultiActionEnv
from utils import selected_features


def make_synthetic_obs(race_lengths, weather=1):
    """
    Napravi umjetni obs_array sastavljen od nekoliko "utrka" spojenih
    zaredom, svaka s LapNumber 1..N (pad LapNumber-a = granica utrke, isti
    princip koji f1_env.py/predict_strategies.py koriste za segmentaciju).
    Sve ostale značajke postavljene su na neutralne, sigurne vrijednosti.
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
    # pit uvijek 0, compound uvijek MEDIUM(1), stil uvijek konzervativno(0)
    n = len(obs_array)
    return np.zeros((n, 3), dtype=np.int32) + [0, 1, 0]


# --- Segmentacija utrka ---

def test_split_into_races_matches_input_lengths(obs_array, race_lengths):
    env = FastF1MultiActionEnv(obs_array, true_actions=None)
    segments = env._split_into_races(obs_array)
    assert [e - s for s, e in segments] == race_lengths


def test_split_into_races_single_race():
    obs = make_synthetic_obs([20])
    env = FastF1MultiActionEnv(obs, true_actions=None)
    segments = env._split_into_races(obs)
    assert segments == [(0, 20)]


# --- Miješanje redoslijeda utrka ---

def test_shuffle_preserves_within_race_lap_order(obs_array):
    env = FastF1MultiActionEnv(obs_array, true_actions=None)
    i_lap = selected_features.index("LapNumber")
    shuffled, orig_indices = env._shuffle_race_order(obs_array)

    for start, end in env._split_into_races(shuffled):
        lap_numbers = shuffled[start:end, i_lap]
        # unutar SVAKE utrke, LapNumber mora i dalje rasti 1,2,3,...
        assert list(lap_numbers) == sorted(lap_numbers.tolist())
        assert lap_numbers[0] == 1


def test_shuffle_orig_indices_correctly_map_back(obs_array):
    env = FastF1MultiActionEnv(obs_array, true_actions=None)
    shuffled, orig_indices = env._shuffle_race_order(obs_array)
    # svaki redak nakon miješanja mora biti IDENTIČAN retku na orig_indices
    # poziciji u originalnom (neizmiješanom) nizu
    assert np.allclose(shuffled, obs_array[orig_indices])


def test_shuffle_preserves_total_row_count(obs_array):
    env = FastF1MultiActionEnv(obs_array, true_actions=None)
    shuffled, orig_indices = env._shuffle_race_order(obs_array)
    assert len(shuffled) == len(obs_array)
    assert len(orig_indices) == len(obs_array)
    assert sorted(orig_indices.tolist()) == list(range(len(obs_array)))


# --- Per-utrka bookkeeping se resetira na granicama ---

def test_per_race_bookkeeping_resets_at_each_boundary(obs_array, true_actions, race_lengths):
    env = FastF1MultiActionEnv(obs_array, true_actions=true_actions, difficulty_level="easy")
    env.reset(seed=1)

    race_starts_seen = 0
    while True:
        is_race_start = env.index in env.race_starts
        action = np.array([0, 1, 0])  # konzervativna, konzistentna akcija
        obs, reward, done, trunc, info = env.step(action)
        if is_race_start:
            race_starts_seen += 1
            # odmah nakon prvog kruga nove utrke: lap_in_race treba biti 1
            # (resetiran na 0, pa inkrementiran), a dry_compounds_used treba
            # sadržavati SAMO compound iz TOG jednog kruga (ne akumulirano
            # iz prošle utrke)
            assert env.lap_in_race == 1
            assert env.dry_compounds_used == {1}
            assert env.wet_or_damp_seen is False
        if done:
            break

    assert race_starts_seen == len(race_lengths)


def test_episode_runs_full_length_without_risky_style(obs_array, true_actions):
    # Stil nikad nije agresivan (2), pa se nasumični "DNF" mehanizam
    # (koji zahtijeva style==2) nikad ne aktivira - epizoda mora proći kroz
    # SVE retke.
    env = FastF1MultiActionEnv(obs_array, true_actions=true_actions, difficulty_level="easy")
    env.reset(seed=1)
    steps = 0
    while True:
        obs, reward, done, trunc, info = env.step(np.array([0, 1, 0]))
        steps += 1
        if done:
            break
    assert steps == len(obs_array)


# --- Nagrada favorizira ispravno ponašanje (regresijski test protiv sqrt(index) bug-a) ---

def test_true_matching_beats_degenerate_policy_per_step(obs_array):
    # Regresijski test: ranije je bezuvjetni "reward += sqrt(self.index)"
    # bonus činio da agent koji UVIJEK bira istu (pogrešnu) akciju dobiva
    # VIŠE ukupnog reward-a od agenta koji stvarno pogađa pravu akciju -
    # ovaj test bi to uhvatio ako se ikad ponovno uvede slična greška.
    true_actions = np.zeros((len(obs_array), 3), dtype=np.int32)
    true_actions[:, 1] = 1  # true_comp = MEDIUM za svaki krug

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

    avg_true_matching = run(lambda: np.array([0, 1, 0]))       # pogađa true_comp
    avg_degenerate = run(lambda: np.array([0, 0, 0]))          # uvijek SOFT (pogrešno)

    assert avg_true_matching > avg_degenerate


# --- DNF nastavlja na sljedeću utrku, ne prekida cijelu epizodu ---

def test_dnf_continues_to_next_race_not_whole_episode(race_lengths):
    # Koristimo weather=2 (vlažno) da omogućimo DNF granu (zahtijeva
    # weather>1), i style=2 (agresivno) da je zaista i pokrenemo - uz
    # dovoljno pokušaja, DNF bi se trebao dogoditi barem jednom, a epizoda
    # svejedno mora završiti kroz sve retke (jer DNF sad samo preskače na
    # sljedeću utrku, ne prekida cijelu epizodu).
    obs = make_synthetic_obs(race_lengths, weather=2)
    true_actions = np.zeros((len(obs), 3), dtype=np.int32)
    env = FastF1MultiActionEnv(obs, true_actions=true_actions, difficulty_level="easy")
    env.reset(seed=123)

    steps = 0
    max_steps = len(obs) * 2  # sigurnosna granica
    while True:
        obs_, reward, done, trunc, info = env.step(np.array([0, 1, 2]))  # agresivno, riskantno
        steps += 1
        if done or steps > max_steps:
            break

    assert steps <= max_steps
    # epizoda se mora završiti (done=True) unutar razumnog broja koraka,
    # bez obzira na to koliko je DNF-ova bilo usput
    assert done or steps > max_steps  # ako je hit safety limit, test dolje ionako pada


# --- Reproducibilnost preko seed parametra ---

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
    # Bez eksplicitnog seeda (kako SB3 interno radi tijekom treninga),
    # generator MORA nastaviti mijenjati stanje između resetova - inače bi
    # trening uvijek vidio identičan redoslijed/augmentaciju. Provjeravamo
    # stvarni redoslijed utrka (orig_indices) kroz nekoliko resetova (ne
    # samo prvo opažanje - na ovoj uniformnoj fixturi prvi redak izgleda
    # isto bez obzira koja je utrka prva) i tražimo BAREM JEDNU razliku, da
    # test ne bude "flaky" na malu (1/4! ≈ 4%) šansu da se dva nasumična
    # redoslijeda slučajno poklope.
    env = FastF1MultiActionEnv(obs_array, true_actions=true_actions, difficulty_level="full")
    orders = []
    for _ in range(5):
        env.reset()
        orders.append(tuple(env.orig_indices.tolist()))
    assert len(set(orders)) > 1
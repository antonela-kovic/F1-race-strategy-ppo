# f1_env.py
# Gym environment for F1 race-strategy decisions (pit stop, tyre compound,
# driving style), built on real FastF1 lap data and augmented with synthetic
# rain and Safety Car / VSC scenarios.
import gym
from gym import spaces
import numpy as np
from utils import selected_features, scale_observation

# Curriculum difficulty does not filter laps by weather (all available races
# are dry). Instead, it controls how often the agent encounters synthetic but
# temporally consistent (block-wise, not per-lap) rain/damp and Safety Car/VSC
# scenarios injected on top of the real dry data. Goal: an agent that can
# handle any race, not only the three dry races that are actually available.
DIFFICULTY_AUGMENT_PARAMS = {
    "easy":   {"p_weather_event": 0.015, "p_caution_event": 0.010},
    "medium": {"p_weather_event": 0.040, "p_caution_event": 0.025},
    "full":   {"p_weather_event": 0.070, "p_caution_event": 0.040},
}

class FastF1MultiActionEnv(gym.Env):
    def __init__(self, obs_array, true_actions=None, difficulty_level="easy"):
        super().__init__()
        self.obs_array = obs_array
        self.true_actions = true_actions
        self.difficulty_level = difficulty_level
        self.index = 0
        self.previous_actions = []

        self.action_space = spaces.MultiDiscrete([2, 5, 3])  # pit, compound, style
        # After scale_observation(), every feature is divided by its scale, so
        # the observed range is roughly [0, ~1.5]; the upper bound of 2.0 leaves
        # a small margin for edge cases and augmented values.
        self.observation_space = spaces.Box(
            low=0, high=2.0, shape=(obs_array.shape[1],), dtype=np.float32
        )

        self._i_weather = selected_features.index("Weather")
        self._i_trackstatus = selected_features.index("TrackStatus")
        self._i_lapnum = selected_features.index("LapNumber")

        # Single random generator used for ALL stochasticity in this class
        # (race-order shuffling, synthetic augmentation, random DNF).
        # See reset() for how seeding works.
        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        # Seeding: an explicit seed makes reset() fully deterministic (same race
        # order, same synthetic augmentation, same DNF outcomes), which keeps
        # evaluation runs in train.py comparable across episodes. Without a
        # seed (as SB3 calls it during training), the generator continues
        # from its current state and episodes stay random.
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.index = 0
        self.previous_actions = []

        shuffled, orig_indices = self._shuffle_race_order(self.obs_array)
        augmented, synth_weather, synth_caution = self._augment_scenarios(shuffled)
        self.filtered_obs_array = augmented
        # Maps a position in filtered_obs_array to a position in self.true_actions.
        # Required because races are shuffled; without it the ground-truth reward
        # would be compared against the wrong lap.
        self.orig_indices = orig_indices
        self.is_synthetic_weather = synth_weather
        self.is_synthetic_caution = synth_caution

        # Race boundaries inside the (shuffled) array for this episode. The array
        # is N concatenated real races, not one long race, so all per-race
        # bookkeeping (tyre/weather history, monotony checks, terminal penalties)
        # must be reset at EVERY boundary, not only at the end of the array.
        self.race_segments = self._split_into_races(self.filtered_obs_array)
        self.race_starts = set(s for s, _ in self.race_segments)
        self.race_end_for = np.empty(len(self.filtered_obs_array), dtype=np.int64)
        for s, e in self.race_segments:
            self.race_end_for[s:e] = e - 1

        # Tracks whether the race has been dry so far and which dry compounds
        # were used. Needed for the mandatory two-dry-compounds rule (the same
        # rule safety_layer.py enforces as post-processing; here it is also
        # taught through the reward so the RAW policy approaches SAFE behaviour).
        # All of this is per race and is also reset at race boundaries in step().
        self.wet_or_damp_seen = False
        self.dry_compounds_used = set()
        self.lap_in_race = 0
        return scale_observation(self.filtered_obs_array[self.index]), {}

    def _split_into_races(self, array):
        """Return a list of (start, end) index ranges, one per race/stint in the
        array, detected by a drop in LapNumber (a drop means a new race or
        driver). Same segmentation principle as in predict_strategies.py."""
        lap_numbers = array[:, self._i_lapnum]
        starts = [0]
        for idx in range(1, len(lap_numbers)):
            if lap_numbers[idx] < lap_numbers[idx - 1]:
                starts.append(idx)
        starts.append(len(array))
        return [(starts[k], starts[k + 1]) for k in range(len(starts) - 1)]

    def _shuffle_race_order(self, array):
        """Shuffle the ORDER of whole races (not individual laps within them).
        This preserves the real lap sequence inside each race, which the
        synthetic blocks below and TyreLife/Stint progression rely on, while
        still varying the training order from episode to episode.

        Also returns orig_indices, the mapping to the original indices in
        obs_array, because self.true_actions is indexed in the ORIGINAL
        (unshuffled) order."""
        races = self._split_into_races(array)
        order = list(range(len(races)))
        self._rng.shuffle(order)
        shuffled = np.concatenate([array[races[k][0]:races[k][1]] for k in order], axis=0)
        orig_indices = np.concatenate([np.arange(races[k][0], races[k][1]) for k in order])
        return shuffled, orig_indices

    def increase_difficulty(self):
        if self.difficulty_level == "easy":
            self.difficulty_level = "medium"
        elif self.difficulty_level == "medium":
            self.difficulty_level = "full"

    def _augment_scenarios(self, array):
        """
        Return a COPY of the array with synthetic but temporally consistent
        (block-wise) rain/damp and Safety Car/VSC scenarios injected per race
        (a block never crosses the boundary between two real races).

        Also returns two boolean masks marking which laps were synthetically
        modified. step() uses them to DISABLE the ground-truth (real driver)
        reward on those laps: the real driver never raced under these
        invented conditions, so their real (dry) decision is no longer a valid
        learning target there. On those laps the agent relies only on the
        rule-based reward terms.
        """
        params = DIFFICULTY_AUGMENT_PARAMS.get(self.difficulty_level, DIFFICULTY_AUGMENT_PARAMS["full"])
        arr = array.copy()
        n = arr.shape[0]
        synth_weather = np.zeros(n, dtype=bool)
        synth_caution = np.zeros(n, dtype=bool)

        for start, end in self._split_into_races(arr):
            # --- persistent rain/damp blocks ---
            idx = start
            while idx < end:
                if self._rng.random() < params["p_weather_event"]:
                    block_len = min(int(self._rng.integers(5, 15)), end - idx)
                    is_wet = self._rng.random() < 0.35
                    arr[idx:idx + block_len, self._i_weather] = 3 if is_wet else 2
                    synth_weather[idx:idx + block_len] = True
                    idx += block_len
                else:
                    idx += 1

            # --- occasional Safety Car / VSC / yellow flag (short blocks) ---
            idx = start
            while idx < end:
                if self._rng.random() < params["p_caution_event"]:
                    block_len = min(int(self._rng.integers(2, 6)), end - idx)
                    code = self._rng.choice([2, 4, 6])  # yellow / Safety Car / VSC
                    arr[idx:idx + block_len, self._i_trackstatus] = code
                    synth_caution[idx:idx + block_len] = True
                    idx += block_len
                else:
                    idx += 1

        return arr, synth_weather, synth_caution

    def step(self, action):
        # A new race can start INSIDE the same episode (filtered_obs_array is N
        # concatenated real races), so reset all per-race bookkeeping to avoid
        # mixing races (e.g. the monotony check comparing the last lap of race A
        # with the driving style from race B).
        if self.index in self.race_starts and self.index != 0:
            self.wet_or_damp_seen = False
            self.dry_compounds_used = set()
            self.previous_actions = []
            self.lap_in_race = 0

        pit, comp, style = action
        self.previous_actions.append(action)

        lap = self.filtered_obs_array[self.index]
        i = lambda name: selected_features.index(name)
        lap_time = lap[i("LapTime")]
        tyre_life = lap[i("TyreLife")]
        front_gap = lap[i("FrontGap")]
        weather = lap[i("Weather")]
        track_status = lap[i("TrackStatus")] if "TrackStatus" in selected_features else 1
        is_synth_weather_lap = bool(self.is_synthetic_weather[self.index])
        is_synth_caution_lap = bool(self.is_synthetic_caution[self.index])

        # Snapshot BEFORE the update, needed for the "first use of this compound" bonus
        self._dry_compounds_before_this_lap = set(self.dry_compounds_used)

        # Update history for the two-dry-compounds rule
        if weather in (2, 3):
            self.wet_or_damp_seen = True
        if comp in (0, 1, 2):
            self.dry_compounds_used.add(int(comp))

        # Base reward, normalised to keep its scale comparable to the decision terms
        reward = (100 - lap_time * 0.5 + 0.3 * tyre_life) / 10.0

        # Pit logic
        if pit == 1:
            reward += 1.5 if (tyre_life < 10 or weather > 1) else -1.0
        if pit == 0 and tyre_life < 5:
            reward -= 3.0

        # Driving style
        if style == 0 and front_gap < 1.5:
            reward += 2.0
        if style == 2 and tyre_life > 20:
            reward -= 1.5

        # === Safety Car / VSC / red / yellow flag (TrackStatus) ===
        # FastF1 codes: 1=clear, 2=yellow, 4=SC, 5=red, 6=VSC, 7=VSC ending
        caution = track_status in (2, 4, 5, 6, 7)
        if caution and style == 2:
            reward -= 2.0  # overtaking is not allowed under caution
        if track_status in (4, 6) and pit == 1:
            reward += 1.5  # "cheap" pit window under SC/VSC, a realistic incentive
        if track_status == 5 and pit == 1:
            reward += 1.0  # red flag: tyre change without the usual time loss

        # === Compound evaluation ===
        # Only clearly wrong choices are penalised here (e.g. INTER/WET on a dry
        # track). The choice between SOFT, MEDIUM and HARD is left to the match
        # with the real driver's tyre (true_comp, below) and to the dry-compound
        # variety incentive, because real F1 strategy requires alternating dry
        # compounds instead of always rewarding the same one.
        if weather == 1 and comp in (3, 4):
            reward -= 2.5  # INTER/WET on a dry track makes no sense
        elif weather == 2 and comp != 3:
            reward -= 1.5
        elif weather == 3 and comp != 4:
            reward -= 2.5

        # Additional penalty for the wrong compound in rain
        if weather == 3 and comp != 4:
            reward -= 2.5

        # Dense (per-step) incentive for real dry-compound variety during a dry
        # race. If the agent introduces a NEW (not yet used) dry compound while
        # the tyres are fairly worn, it is most likely a genuine strategic change,
        # so it is rewarded immediately instead of relying only on the terminal
        # penalty at the end of the race.
        if (
            not self.wet_or_damp_seen
            and comp in (0, 1, 2)
            and comp not in self._dry_compounds_before_this_lap
            and tyre_life < 15
        ):
            reward += 2.5

        # Compound change in bad weather: compare with the PREVIOUS lap's action
        # (previous_actions[-1] is the current action appended above).
        if len(self.previous_actions) > 1 and weather > 1:
            if comp != self.previous_actions[-2][1]:
                reward += 1.0
            else:
                reward -= 0.5

        # Note: there is intentionally no unconditional per-lap "survival" bonus.
        # Such a term grew with the lap index, dominated the total episode reward
        # and taught the agent to simply survive longer instead of learning
        # state-dependent strategy.

        # Penalty if the agent never pitted / did not use two dry compounds.
        # Checked at the end of EVERY race inside the episode (race_end_for),
        # not only at the end of the whole concatenated array.
        if self.index == self.race_end_for[self.index]:
            if 1 not in [a[0] for a in self.previous_actions]:
                reward -= 7.0

            # Penalty if THIS race was dry and the agent used only ONE dry
            # compound until the end (FIA: two dry compounds are mandatory in a
            # dry race). safety_layer.py later enforces the same rule; here the
            # agent learns to avoid the violation on its own.
            if not self.wet_or_damp_seen and len(self.dry_compounds_used) < 2:
                reward -= 8.0

        # Random penalty for risky behaviour: if triggered, the agent "retires"
        # (DNF) from THIS race, but the episode continues with the NEXT race
        # instead of ending, so later races in the episode are still reached.
        if style == 2 and weather > 1 and self._rng.random() < 0.1:
            reward -= 10.0
            race_end = self.race_end_for[self.index]
            if race_end >= len(self.filtered_obs_array) - 1:
                # this was the last race in the episode, so the episode ends
                return scale_observation(self.filtered_obs_array[self.index]), reward, True, False, {}
            # DNF in this race: jump to the start of the NEXT race, the episode continues
            self.index = race_end + 1
            self.wet_or_damp_seen = False
            self.dry_compounds_used = set()
            self.previous_actions = []
            self.lap_in_race = 0
            return scale_observation(self.filtered_obs_array[self.index]), reward, False, False, {}

        # Strategy monotony penalty. Uses lap_in_race (a per-race counter, not the
        # global self.index) so laps from two different races are never compared
        # right after a race boundary.
        if self.lap_in_race > 3:
            recent = self.previous_actions[-3:]
            if len(set([x[1] for x in recent])) == 1:
                reward -= 3.0
            if len(set([x[2] for x in recent])) == 1:
                reward -= 1.5
            if len(set([x[0] for x in recent])) == 1:
                reward -= 1.0

        if self.lap_in_race > 5:
            recent_styles = [x[2] for x in self.previous_actions[-5:]]
            if len(set(recent_styles)) == 1:
                reward -= 3.0

        if self.lap_in_race > 10:
            last_10 = self.previous_actions[-10:]
            unique_strategies = len(set(tuple(x) for x in last_10))
            if unique_strategies >= 5:
                reward += 3.0

        # Additional weather-dependent penalties
        if weather == 2:
            if comp != 3:
                reward -= 1.5
            if style == 2:
                reward -= 1.5

        # Imitation of the real driver's decisions.
        # self.true_actions is indexed in the ORIGINAL (unshuffled) order of
        # obs_array, while self.index is a position in the SHUFFLED
        # filtered_obs_array, so the lookup goes through self.orig_indices.
        true_idx = self.orig_indices[self.index]
        if self.true_actions is not None and true_idx < len(self.true_actions):
            true_pit, true_comp, true_style = self.true_actions[true_idx]
            # Real pit stops are rare (~2.5% of laps). A symmetric reward made
            # "always predict no pit" the most profitable strategy, so the pit
            # reward is asymmetric (+4/-3 for pit laps vs +1/-1.5 otherwise): it
            # favours catching the rare pit moment while still penalising false
            # positives enough that pitting "just in case" does not pay off.
            if true_pit == 1:
                reward += 4.0 if pit == 1 else -3.0
            else:
                reward += 1.0 if pit == 0 else -1.5
            # On laps synthetically turned into rain/damp, the real driver never
            # raced under those conditions, so their real (dry) tyre is not a
            # valid target; the rule-based terms above handle those laps instead.
            if not is_synth_weather_lap:
                reward += 4.0 if comp == true_comp else -1.5
            if not (is_synth_weather_lap or is_synth_caution_lap):
                reward += 4.0 if style == true_style else -1.5

        # Small constant per-lap penalty
        reward -= 0.5

        self.lap_in_race += 1

        # End of episode
        done = self.index >= len(self.filtered_obs_array) - 1
        if not done:
            self.index += 1
            obs = self.filtered_obs_array[self.index]
        else:
            obs = self.filtered_obs_array[-1]

        return scale_observation(obs), reward, done, False, {}

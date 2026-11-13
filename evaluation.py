# evaluation.py
# Shared evaluation logic: runs a policy over real laps race by race, applies
# the safety layer and computes metrics against the real drivers' decisions.
# Used by predict_strategies.py (all data) and evaluate_holdout.py (held-out
# track). Kept free of plotting and file I/O so it can be unit-tested.
import numpy as np

from safety_layer import RaceState, verify_action
from utils import STYLE_UNLABELED, scale_observation, selected_features


def split_into_races(obs_array):
    """Return [(start, end), ...] for every race/driver block, detected by a
    drop in LapNumber (same principle as f1_env.py)."""
    lap_numbers = obs_array[:, selected_features.index("LapNumber")]
    starts = [0]
    for idx in range(1, len(lap_numbers)):
        if lap_numbers[idx] < lap_numbers[idx - 1]:
            starts.append(idx)
    starts.append(len(obs_array))
    return [(starts[k], starts[k + 1]) for k in range(len(starts) - 1)]


def select_tracks(obs_array, true_actions, track_ids, exclude=None, only=None):
    """Return the rows of obs_array/true_actions whose track is not `exclude`
    (training set) or equal to `only` (test set). Whole races are kept intact,
    so race segmentation by LapNumber still works on the result."""
    track_ids = np.asarray(track_ids)
    if exclude is not None:
        mask = track_ids != exclude
    elif only is not None:
        mask = track_ids == only
    else:
        mask = np.ones(len(track_ids), dtype=bool)
    if not mask.any():
        raise ValueError(f"No laps selected (exclude={exclude}, only={only}); "
                         f"tracks in data: {sorted(set(track_ids))}")
    return obs_array[mask], true_actions[mask]


def run_policy(model, obs_array, force_window=3):
    """
    Predict RAW actions with `model` and SAFE actions with the safety layer for
    every lap. A new RaceState is created for every race so R10 is checked at
    the end of each race, not only at the end of the concatenated array.
    Returns (raw_actions, safe_actions, stats).
    """
    i_weather = selected_features.index("Weather")
    raw_actions, safe_actions = [], []
    stats = {"pit_overrides": 0, "comp_overrides": 0, "style_overrides": 0,
             "r10_events": 0, "laps_with_violation": 0, "laps_with_override": 0,
             "dry_races": 0, "dry_races_r10_ok": 0}

    for start, end in split_into_races(obs_array):
        race_state = RaceState(total_laps=end - start, force_window=force_window)
        for obs in obs_array[start:end]:
            action, _ = model.predict(scale_observation(obs).reshape(1, -1), deterministic=True)
            raw = np.asarray(action, dtype=np.int32).reshape(-1)[:3]
            # verify_action receives the UNSCALED state (thresholds in real units)
            safe, report = verify_action(obs, raw, selected_features, race_state=race_state)

            for o in report.overrides:
                key = {"pit": "pit_overrides", "compound": "comp_overrides",
                       "style": "style_overrides"}.get(o["field"])
                if key:
                    stats[key] += 1
                if o["rule"] == "R10_MANDATORY_TWO_DRY_COMPOUNDS":
                    stats["r10_events"] += 1
            stats["laps_with_violation"] += int(len(report.violations_before) > 0)
            stats["laps_with_override"] += int(len(report.overrides) > 0)

            race_state.record(weather=int(round(float(obs[i_weather]))), compound=int(safe[1]))
            raw_actions.append(raw)
            safe_actions.append(np.asarray(safe, dtype=np.int32))

        if race_state.is_dry_race_so_far:
            stats["dry_races"] += 1
            stats["dry_races_r10_ok"] += int(len(race_state.dry_compounds_used) >= 2)

    return np.array(raw_actions, dtype=np.int32), np.array(safe_actions, dtype=np.int32), stats


def majority_class(labels, ignore=None):
    labels = np.asarray(labels)
    if ignore is not None:
        labels = labels[labels != ignore]
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def compute_metrics(true_actions, pred_actions, majority=None):
    """
    Metrics for one set of predictions.
    - compound: accuracy on all laps
    - style: accuracy only on laps with a pace-based label (STYLE_UNLABELED
      laps are skipped, see utils.compute_style_labels)
    - pit: accuracy, recall, precision and predicted pit rate (pit stops are
      ~2.5 % of laps, so accuracy alone is not meaningful)
    `majority` = (pit, compound, style) majority classes, ideally taken from
    the TRAINING data; the baseline always predicts them.
    """
    true_actions = np.asarray(true_actions)
    pred_actions = np.asarray(pred_actions)
    t_pit, t_comp, t_style = true_actions[:, 0], true_actions[:, 1], true_actions[:, 2]
    p_pit, p_comp, p_style = pred_actions[:, 0], pred_actions[:, 1], pred_actions[:, 2]
    labeled = t_style != STYLE_UNLABELED

    tp = int(((p_pit == 1) & (t_pit == 1)).sum())
    m = {
        "n_laps": int(len(true_actions)),
        "n_style_labeled": int(labeled.sum()),
        "compound_acc": float((p_comp == t_comp).mean()),
        "style_acc": float((p_style[labeled] == t_style[labeled]).mean()) if labeled.any() else float("nan"),
        "pit_acc": float((p_pit == t_pit).mean()),
        "pit_recall": tp / int(t_pit.sum()) if t_pit.sum() else float("nan"),
        "pit_precision": tp / int(p_pit.sum()) if p_pit.sum() else float("nan"),
        "pit_rate": float(p_pit.mean()),
        "true_pits": int(t_pit.sum()),
    }
    if majority is not None:
        mp, mc, ms = majority
        m["baseline_compound_acc"] = float((t_comp == mc).mean())
        m["baseline_style_acc"] = float((t_style[labeled] == ms).mean()) if labeled.any() else float("nan")
        m["baseline_pit_acc"] = float((t_pit == mp).mean())
    return m

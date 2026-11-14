# test_evaluation.py
# Unit tests for evaluation.py: track split for the held-out evaluation and
# the metrics (style only on labeled laps, pit recall/precision).
import numpy as np
import pytest

from evaluation import compute_metrics, majority_class, select_tracks, split_into_races
from test_f1_env import make_synthetic_obs
from utils import STYLE_UNLABELED


def test_select_tracks_exclude_and_only_are_disjoint_and_complete():
    obs = make_synthetic_obs([5, 6, 7])
    true = np.zeros((len(obs), 3), dtype=np.int32)
    tracks = np.array(["A"] * 5 + ["B"] * 6 + ["C"] * 7)
    train_obs, _ = select_tracks(obs, true, tracks, exclude="B")
    test_obs, _ = select_tracks(obs, true, tracks, only="B")
    assert len(train_obs) == 12 and len(test_obs) == 6
    # whole races are kept, so segmentation still finds them
    assert [e - s for s, e in split_into_races(train_obs)] == [5, 7]
    assert [e - s for s, e in split_into_races(test_obs)] == [6]


def test_select_tracks_unknown_track_raises():
    obs = make_synthetic_obs([5])
    with pytest.raises(ValueError):
        select_tracks(obs, np.zeros((5, 3)), np.array(["A"] * 5), only="Z")


def test_metrics_ignore_unlabeled_style_laps():
    true = np.array([[0, 1, 2], [0, 1, STYLE_UNLABELED], [1, 2, 0], [0, 1, STYLE_UNLABELED]])
    pred = np.array([[0, 1, 2], [0, 1, 1], [0, 2, 1], [1, 0, 0]])
    m = compute_metrics(true, pred, majority=(0, 1, 2))
    assert m["n_style_labeled"] == 2
    assert m["style_acc"] == pytest.approx(0.5)
    assert m["compound_acc"] == pytest.approx(0.75)
    assert m["pit_recall"] == 0.0 and m["pit_precision"] == 0.0
    assert m["baseline_style_acc"] == pytest.approx(0.5)
    assert m["baseline_pit_acc"] == pytest.approx(0.75)


def test_majority_class_ignores_unlabeled():
    assert majority_class([STYLE_UNLABELED] * 5 + [1, 1, 0], ignore=STYLE_UNLABELED) == 1


def test_seed_summary_pools_pits_over_folds_and_counts_blocked_stops():
    import pandas as pd
    from evaluate_holdout import seed_summary
    rows = []
    for track, true_pits, n_laps in (("A", 5, 200), ("B", 4, 200)):
        # RAW catches every real stop, SAFE misses one at track A
        for variant, recall, rate in (("RAW", 1.0, 0.05), ("SAFE", 0.8 if track == "A" else 1.0, 0.05)):
            rows.append({"seed": 0, "held_out_track": track, "split": "test (held-out)",
                         "variant": variant, "compound_acc": 0.5, "style_acc": 0.4,
                         "pit_recall": recall, "true_pits": true_pits, "pit_rate": rate,
                         "n_laps": n_laps,
                         "overrides": "R5_BLOCK_PIT_ON_FRESH_TYRES=2" if variant == "SAFE" else None})
    s = seed_summary(pd.DataFrame(rows)).iloc[0]
    assert s.pits_caught == 8 and s.true_pits == 9
    assert s.pits_blocked_by_layer == 1
    assert s.pit_precision == pytest.approx(8 / 20)
    assert "R5_BLOCK_PIT_ON_FRESH_TYRES=4" in s.overrides

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

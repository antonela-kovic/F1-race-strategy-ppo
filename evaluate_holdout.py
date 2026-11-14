# evaluate_holdout.py
# Leave-one-track-out evaluation: for every track, train on the OTHER tracks
# and evaluate on the held-out one, which the model has never seen (neither in
# training nor in checkpoint selection). This answers "does the agent
# generalise to a new circuit?", which the in-sample numbers of
# predict_strategies.py cannot.
#
#   python evaluate_holdout.py                  # all folds, seeds 0 1 2 (train + evaluate)
#   python evaluate_holdout.py --tracks Monza   # one fold
#   python evaluate_holdout.py --seeds 0        # a single training seed (faster)
#   python evaluate_holdout.py --skip-training  # reuse saved holdout models
#
# Training is not deterministic enough to trust one run: with the same code,
# compound accuracy on one held-out track moved by up to ~20 points between
# training seeds. The report therefore gives the MEAN over several seeds, the
# min-max range, and one summary row per seed.
#
# Results: data/holdout_results.csv and docs/holdout_results.md
import argparse
import os

import numpy as np
import pandas as pd

from evaluation import compute_metrics, majority_class, run_policy, select_tracks
from utils import STYLE_UNLABELED


def majority_from(true_actions):
    return (majority_class(true_actions[:, 0]),
            majority_class(true_actions[:, 1]),
            majority_class(true_actions[:, 2], ignore=STYLE_UNLABELED))


def evaluate_fold(test_track, obs_array, true_actions, track_ids, args, seed):
    from stable_baselines3 import PPO
    from train import holdout_model_paths, train_agent

    train_obs, train_true = select_tracks(obs_array, true_actions, track_ids, exclude=test_track)
    test_obs, test_true = select_tracks(obs_array, true_actions, track_ids, only=test_track)
    train_tracks = sorted(set(track_ids) - {test_track})
    best_path, final_path = holdout_model_paths(test_track, seed)

    print(f"\n=== Seed {seed}, fold: train on {train_tracks} ({len(train_obs)} laps), "
          f"test on {test_track} ({len(test_obs)} laps) ===")
    if not (args.skip_training and os.path.exists(best_path)):
        train_agent(train_obs, train_true, best_path, final_path,
                    n_iterations=args.iterations, steps_per_iteration=args.steps,
                    seed=seed, verbose=0)
    model = PPO.load(best_path)

    # Majority-class baseline is taken from the TRAINING tracks: that is the
    # best a model without any state information could know in advance.
    majority = majority_from(train_true)
    rows = []
    for split, obs, true in (("test (held-out)", test_obs, test_true),
                             ("train (in-sample)", train_obs, train_true)):
        raw, safe, stats = run_policy(model, obs)
        for variant, pred in (("RAW", raw), ("SAFE", safe)):
            m = compute_metrics(true, pred, majority)
            m.update({"seed": seed, "held_out_track": test_track,
                      "train_tracks": "+".join(train_tracks), "split": split, "variant": variant})
            if variant == "SAFE":
                m["r10_ok"] = f"{stats['dry_races_r10_ok']}/{stats['dry_races']}"
                m["overrides"] = "; ".join(f"{r}={c}" for r, c in sorted(stats["rule_counts"].items()))
            rows.append(m)
    return rows


def _caught(rows):
    """Real pit stops caught, summed over folds."""
    return int((rows["pit_recall"] * rows["true_pits"]).round().sum())


def seed_summary(df):
    """One row per seed: held-out (test) metrics pooled over the three folds."""
    out = []
    for seed, g in df[df["split"] == "test (held-out)"].groupby("seed"):
        safe, raw = g[g["variant"] == "SAFE"], g[g["variant"] == "RAW"]
        predicted = (safe["pit_rate"] * safe["n_laps"]).round().sum()
        rules = {}
        for text in safe["overrides"].dropna():
            for item in str(text).split("; "):
                if "=" in item:
                    name, count = item.split("=")
                    rules[name] = rules.get(name, 0) + int(count)
        out.append({
            "seed": seed,
            "compound_acc": safe["compound_acc"].mean(),
            "style_acc": safe["style_acc"].mean(),
            "pits_caught": _caught(safe),
            "true_pits": int(safe["true_pits"].sum()),
            "pit_precision": _caught(safe) / predicted if predicted else float("nan"),
            "pit_rate": safe["pit_rate"].mean(),
            "pits_blocked_by_layer": _caught(raw) - _caught(safe),
            "overrides": ", ".join(f"{k}={v}" for k, v in sorted(rules.items())),
        })
    return pd.DataFrame(out)


def to_markdown(df):
    test = df[(df["split"] == "test (held-out)") & (df["variant"] == "SAFE")]
    train = df[(df["split"] == "train (in-sample)") & (df["variant"] == "SAFE")]
    seeds = sorted(test["seed"].unique())
    tracks = list(dict.fromkeys(test["held_out_track"]))
    per_seed = seed_summary(df)
    n_pits = int(per_seed["true_pits"].iloc[0])
    lines = [
        "# Leave-one-track-out results (SAFE = policy + safety layer)",
        "",
        "Generated by `evaluate_holdout.py`. Each model is trained on two tracks and evaluated",
        "on the held-out track it has never seen. Baseline = majority class of the TRAINING",
        "tracks. Style is scored only on laps with a pace-based label.",
        f"Every fold was trained with {len(seeds)} seeds ({', '.join(str(s) for s in seeds)});",
        "the first table gives the mean over seeds and, for compound, the min-max range.",
        "",
        "## Mean over seeds, per held-out track",
        "",
        "| Held-out track | Compound | Baseline | Style | Baseline | Pit recall | Pit precision | Pit rate | Compound in-sample |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def row(name, t, tr):
        # pit recall / precision pooled over folds and seeds (sum of caught
        # stops / sum of real or predicted stops), not a mean of percentages
        per_seed_comp = t.groupby("seed")["compound_acc"].mean()
        caught = _caught(t)
        predicted = (t["pit_rate"] * t["n_laps"]).round().sum()
        precision = f"{caught / predicted:.0%}" if predicted else "-"
        return (f"| {name} | {per_seed_comp.mean():.1%} ({per_seed_comp.min():.0%}-{per_seed_comp.max():.0%}) "
                f"| {t.baseline_compound_acc.mean():.1%} | {t.style_acc.mean():.1%} "
                f"| {t.baseline_style_acc.mean():.1%} | {caught / t.true_pits.sum():.0%} "
                f"| {precision} | {t.pit_rate.mean():.1%} "
                f"| {tr.compound_acc.mean():.1%} |")

    for track in tracks:
        lines.append(row(track, test[test.held_out_track == track], train[train.held_out_track == track]))
    lines.append(row("**mean**", test, train))
    lines += [
        "",
        "## Per seed, pooled over the three held-out tracks",
        "",
        f"| Seed | Compound | Style | Real pit stops caught (of {n_pits}) | Pit precision | Pit rate "
        "| Real stops blocked by the layer | Overrides by rule |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in per_seed.iterrows():
        lines.append(f"| {r.seed} | {r.compound_acc:.1%} | {r.style_acc:.1%} | {r.pits_caught} "
                     f"| {r.pit_precision:.0%} | {r.pit_rate:.1%} | {r.pits_blocked_by_layer} "
                     f"| {r.overrides} |")
    lines.append(f"| **mean** | {per_seed.compound_acc.mean():.1%} | {per_seed.style_acc.mean():.1%} "
                 f"| {per_seed.pits_caught.mean():.1f} | {per_seed.pit_precision.mean():.0%} "
                 f"| {per_seed.pit_rate.mean():.1%} | {per_seed.pits_blocked_by_layer.mean():.1f} | |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", nargs="*", help="held-out tracks to evaluate (default: all)")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="training seeds; results are averaged over them")
    parser.add_argument("--skip-training", action="store_true",
                        help="reuse saved_models/ppo_holdout_<track>_s<seed>_best.zip if it exists")
    args = parser.parse_args()

    obs_array = np.load("data/obs_array.npy")
    true_actions = np.load("data/true_actions.npy")
    if not os.path.exists("data/track_ids.npy"):
        raise SystemExit("data/track_ids.npy is missing - run prepare_data.py first.")
    track_ids = np.load("data/track_ids.npy")

    tracks = args.tracks or list(dict.fromkeys(track_ids))  # keep data order
    rows = []
    for seed in args.seeds:
        for track in tracks:
            rows += evaluate_fold(track, obs_array, true_actions, track_ids, args, seed)

    df = pd.DataFrame(rows)
    os.makedirs("docs", exist_ok=True)
    df.to_csv("data/holdout_results.csv", index=False)
    md = to_markdown(df)
    with open("docs/holdout_results.md", "w") as f:
        f.write(md)
    print("\n" + md)
    print("Saved data/holdout_results.csv and docs/holdout_results.md")

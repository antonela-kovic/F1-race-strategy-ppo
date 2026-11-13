# F1 Race Strategy Agent – PPO with a Rule-Based Safety Layer

A reinforcement learning agent that makes Formula 1 race-strategy decisions – **pit stop, tyre compound and driving style** – on every lap. It is trained with **Proximal Policy Optimization (PPO)** on real lap data from the [FastF1](https://github.com/theOehrly/Fast-F1) library, and every decision is checked by an explicit **rule-based safety layer** that enforces race regulations and records which rule fired and why. The model is served through a **FastAPI** REST API.

The project combines a learned policy with verifiable rules: the policy proposes, the rules check. This pattern is common in safety-relevant automotive software, where a learned component must stay inside explicit constraints.

---

## At a glance

```mermaid
flowchart LR
    D[(FastF1 lap and<br/>weather data)] --> P[prepare_data.py<br/>features + ground truth]
    P --> E[F1 Gym environment<br/>reward, curriculum,<br/>synthetic rain / SC]
    E --> T[PPO training<br/>MLP 256-128-64]
    T --> M[Trained policy]
    M -->|RAW action| S[Safety layer<br/>rules R1-R10]
    S -->|SAFE action +<br/>explanation| A[FastAPI REST API]
```

| Component | What it does |
|---|---|
| `prepare_data.py` | Downloads 2023 races (Silverstone, Monza, Spa) for four drivers (VER, HAM, LEC, PER), joins real weather data, computes real gaps to the cars ahead/behind and a pace-based style label, and builds observations and ground-truth actions |
| `f1_env.py` | Gym environment with a shaped reward, a three-level curriculum and block-wise synthetic rain/damp and Safety Car/VSC scenarios |
| `train.py` | PPO training (Stable-Baselines3), keeps the best checkpoint by reward per step |
| `safety_layer.py` | Ten named rules (wet tyres in rain, no aggressive style under Safety Car, two mandatory dry compounds, …) that override unsafe actions and explain every override |
| `api.py` | REST API: single prediction, batch prediction over a race, rule-based explanation |
| `evaluate_holdout.py` | Leave-one-track-out evaluation: train on two tracks, test on the third |
| `predict_strategies.py` | In-sample evaluation: RAW policy vs SAFE (policy + rules) against the real drivers' decisions |

### Action space

| Action | Values |
|---|---|
| Pit stop | 0 = stay out, 1 = pit |
| Compound | 0 SOFT, 1 MEDIUM, 2 HARD, 3 INTER, 4 WET |
| Driving style | 0 conservative, 1 neutral, 2 aggressive |

### Safety layer rules

| Rule | Condition | Effect |
|---|---|---|
| R1 | rain | compound must be WET |
| R2 | damp track | compound must be INTER |
| R3a | dry track | INTER/WET not allowed |
| R4 | critically low tyre life | pit is forced |
| R5 | dry track, tyres in very good condition | pointless pit is blocked |
| R6 | wet or damp track | no aggressive style |
| R7 | small gap to the car ahead | conservative style |
| R8 | large gap on a dry track | conservative style relaxed to neutral |
| R9 | Safety Car / VSC / red / yellow flag | no aggressive style (overtaking forbidden) |
| R10 | dry race, only one dry compound used near the end | pit and compound change are forced (FIA rule) |

---

## Results

### Held-out evaluation (leave one track out)

The main result is measured on a track the agent has **never seen**: for each
of the three tracks, a model is trained on the other two and evaluated on the
held-out one (three folds). The held-out track is also excluded from
checkpoint selection, and the majority-class baseline is taken from the
training tracks.

```bash
python prepare_data.py       # needs internet (FastF1)
python evaluate_holdout.py   # ~25 min on a laptop CPU, writes docs/holdout_results.md
```

The table in [`docs/holdout_results.md`](docs/holdout_results.md) reports, per
held-out track: compound accuracy, style accuracy (only on laps with a style
label, see below), pit recall / precision / predicted pit rate, and the same
model's in-sample compound accuracy for comparison.

Pit stops happen on only ~2.5 % of laps, so accuracy is not a meaningful
metric for them (a model that never pits reaches 97.5 %); recall and precision
are reported instead.

### Driving-style label

F1 data has no "driving style" label. Earlier versions used a rule over
`FrontGap`, `RearGap` and `TyreLife`, i.e. a function of features the agent
also observes, with random gaps, so style accuracy only measured how well the
network re-learned that rule. The label is now derived from **measured pace**
(`compute_style_labels` in `utils.py`): within each stint, lap time is fitted
as a linear function of tyre age (tyre degradation + fuel burn), and the
residual decides the label: more than 0.2 s faster than the trend = aggressive,
more than 0.2 s slower = conservative, otherwise neutral. Laps where pace says
nothing about the driver's choice (lap 1, pit in/out laps, SC/VSC/yellow,
laps FastF1 marks as inaccurate) are left unlabeled (`-1`) and are excluded
from the style reward and from style accuracy.

### Previous in-sample numbers (v6, historical)

Before these fixes, v6 was evaluated on all 674 laps it was trained on, with
random gaps and the gap-based style rule: compound 69.6 % (baseline 43.9 %),
style 64.2 % (baseline 42.1 %), pit recall 88 % / precision 24 %. These numbers
are in-sample and are kept only for reference; `predict_strategies.py` still
produces this kind of in-sample report.

---

## Prerequisites

- Python 3.10 or newer
- Internet access the first time `prepare_data.py` runs (FastF1 downloads and caches the data)

The trained models (`saved_models/`) and the prepared data (`data/`) are committed, so the API and the evaluation can run without retraining.

> **Regenerate the data after updating.** The committed `data/` was built by an older `prepare_data.py`: it has random `FrontGap`/`RearGap`, the old gap-based style label, undecoded `TrackStatus` strings (e.g. `41`), no `track_ids.npy`, and its "Spa" laps are actually the **Spanish** GP (FastF1 resolved the short name `'Spa'` to Barcelona). Run `python prepare_data.py` and then `python train.py` / `python evaluate_holdout.py` to get data and models consistent with the current code.

## Installation

```bash
git clone https://github.com/antonela-kovic/f1-race-strategy-ppo.git
cd f1-race-strategy-ppo
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes `gym` and `shimmy`, because the environment uses the older `gym` API together with a newer `stable-baselines3`.

## Usage

### 1. Prepare data and train (optional)

```bash
python prepare_data.py   # downloads races and builds data/obs_array.npy, data/true_actions.npy
python train.py          # PPO training with curriculum, saves saved_models/*_best.zip
```

### 2. Evaluate

```bash
python evaluate_holdout.py              # held-out: train on 2 tracks, test on the 3rd (all folds)
python evaluate_holdout.py --tracks Spa # one fold only
python train.py --holdout Spa           # train a single held-out model with plots
python predict_strategies.py            # in-sample report for the model used by the API
```

`predict_strategies.py` prints RAW vs SAFE accuracy, the safety-layer override rate per field and the R10 check per race, on the same races the model was trained on. The same in-sample evaluation is available interactively in `main.ipynb`.

### 3. Run the REST API

```bash
uvicorn api:app --reload
```

- API: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs

| Endpoint | Description |
|---|---|
| `GET /` | Health check |
| `POST /predict_strategy` | Strategy for a single lap state |
| `POST /predict_strategy_batch` | Strategy for a sequence of laps, treated as one race (R10 is checked automatically) |
| `POST /explain_strategy` | Rule-of-thumb explanation of a state, without the model |

The state is a list of 15 features in this order: `TyreLife, CompoundEncoded, LapNumber, TrackStatus, IsAccurate, Position, LapTime, Sector1Time, Sector2Time, FrontGap, RearGap, Stint, TrackTemperature, AirTemperature, Weather`.

Example request for `POST /predict_strategy`:

```json
{"state": [5, 2, 22, 2, 1, 8, 100.5, 35.0, 34.0, 0.5, 1.0, 2, 23.0, 18.0, 2]}
```

More examples:

```json
// Dry track, tyres in good condition
{"state": [20, 1, 12, 2, 1, 5, 91.2, 30.5, 30.1, 1.8, 2.2, 1, 30.0, 22.0, 1]}

// Rain
{"state": [7, 3, 18, 1, 1, 4, 105.0, 36.5, 35.2, 0.8, 1.5, 1, 19.0, 16.0, 3]}

// Hot track, MEDIUM compound
{"state": [15, 1, 9, 1, 1, 3, 95.0, 31.0, 30.0, 2.5, 3.0, 0, 36.0, 28.0, 1]}
```

The response contains the final (SAFE) action and an explanation with the RAW action, human-readable labels, violated rules and every override (shortened example):

```json
{
  "action": [0, 3, 0],
  "explanation": {
    "raw_action": [0, 1, 0],
    "safe_action": [0, 3, 0],
    "logic": {
      "overrides": [
        {"field": "compound", "from": 1, "to": 3, "rule": "R2_DAMP_REQUIRES_INTER", "why": "Damp: INTER is mandatory"}
      ]
    }
  }
}
```

For a batch request, send a list: `[{"state": [...]}, {"state": [...]}]`.

---

## Testing

Unit tests (pytest) cover the parts that do not depend on the FastF1 network or the trained model: the rules in `safety_layer.py`, the environment mechanics in `f1_env.py` (race segmentation, race-order shuffling, per-race bookkeeping, seeding) and the helpers in `utils.py` (including gap computation and style labels) and the metrics / track split in `evaluation.py`.

```bash
pytest -v
```

`prepare_data.py` is not tested directly because it calls the FastF1 API at import time; its pure helper functions (`classify_weather`, `worst_track_status`) live in `utils.py` so they can be tested in isolation.

---

## Project structure

```
├── api.py                  # FastAPI service
├── f1_env.py               # Gym environment: reward, curriculum, synthetic scenarios
├── train.py                # PPO training with evaluation and plots
├── safety_layer.py         # Rules R1-R10, overrides and explanations
├── evaluate_holdout.py     # Leave-one-track-out evaluation (train on 2, test on 3rd)
├── evaluation.py           # Shared evaluation: per-race policy run, metrics, track split
├── predict_strategies.py   # In-sample evaluation RAW vs SAFE, R10 check per race
├── prepare_data.py         # FastF1 data download and feature engineering
├── utils.py                # Features, observation scaling, pure helpers
├── model.py                # LSTM feature extractor (experimental, not used, see below)
├── main.py                 # Runs data preparation, training and the API in sequence
├── main.ipynb              # Interactive evaluation and API demo
├── test_*.py               # Unit tests
├── saved_models/           # Trained models (best and final)
├── data/                   # Prepared observations, ground truth, track ids, predictions
├── docs/                   # Generated held-out results
└── requirements.txt
```

---

## Known limitations

- **Only three tracks.** The leave-one-track-out evaluation has three folds with ~200 laps each, so the held-out numbers have a large variance; more races would make them more reliable.
- **Gap features are end-of-lap intervals.** `FrontGap` and `RearGap` are the real time differences to the car one position ahead / behind at the timing line at the end of the lap, computed from all drivers' lap times (`compute_gaps` in `utils.py`). They are clipped to 10 s (leader, last car, or a missing neighbour also get 10 s), and they do not see a lapped car that is physically in front.
- **Style is still a proxy.** The pace-based label comes from real timing data, but a slow lap can also be caused by traffic or dirty air rather than a deliberate choice. The residual uses the lap's own `LapTime`, which is also an observed feature, so the agent sees part of the information the label is built from.
- **Pit frequency.** The agent pits more often than real drivers (~9 % of laps vs ~2.5 %), a consequence of the asymmetric reward that prevents the opposite failure (never pitting). The safety layer still enforces the strategy rules independently.
- **Not a physical race simulation.** The environment replays real laps with synthetic rain and Safety Car scenarios on top; `TyreLife` and other features do not change as a consequence of the agent's decisions.
- **No memory across laps.** `LSTMFeatureExtractor` in `model.py` is kept as a starting point for future work. Real memory over laps would require `RecurrentPPO` from `sb3-contrib` instead of standard PPO.

---

## Tech stack

Python · Stable-Baselines3 (PPO) · Gym · PyTorch · FastF1 · pandas · NumPy · scikit-learn · FastAPI · Pydantic · pytest · Matplotlib

## Author

Antonela Ković – [GitHub](https://github.com/antonela-kovic)

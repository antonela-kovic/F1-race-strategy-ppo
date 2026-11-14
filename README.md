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
| `prepare_data.py` | Downloads 2023 races (Silverstone, Monza, Barcelona) for four drivers (VER, HAM, LEC, PER), joins real weather data, computes real gaps to the cars ahead/behind and a pace-based style label, and builds observations and ground-truth actions |
| `f1_env.py` | Gym environment with a shaped reward, a three-level curriculum and block-wise synthetic rain/damp and Safety Car/VSC scenarios |
| `train.py` | PPO training (Stable-Baselines3), keeps the best checkpoint by reward per step |
| `safety_layer.py` | Nine named rules (wet tyres in rain, no aggressive style under Safety Car, two mandatory dry compounds, …) that override unsafe actions and explain every override |
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
| R4 | *(removed)* | an earlier version forced a pit on brand-new tyres, see below |
| R5 | dry track, no SC/VSC/red flag, tyres younger than 10 laps | pointless pit is blocked |
| R6 | wet or damp track | no aggressive style |
| R7 | small gap to the car ahead | conservative style |
| R8 | large gap on a dry track | conservative style relaxed to neutral |
| R9 | Safety Car / VSC / red / yellow flag | no aggressive style (overtaking forbidden) |
| R10 | dry race, only one dry compound used near the end | pit and compound change are forced (FIA rule) |

### Tyre age (`TyreLife`)

In FastF1, `TyreLife` is the **age** of the tyre set in laps (1 = new tyre).
Earlier versions of the safety layer and the reward read it as *remaining*
life: R4 forced a pit when `TyreLife < 2` (on brand-new tyres, 22 laps in the
data), R5 blocked a pit when `TyreLife > 30` (on the oldest tyres; it blocked
the two real stops at 33 laps, both under the Safety Car at Silverstone), and
the reward paid for pitting on new tyres. All of this now reads `TyreLife` as
age, with thresholds taken from the 17 real pit stops (`TYRE_AGE_FRESH` and
`TYRE_AGE_WORN` in `utils.py`):

| Real pit stops (2023 data) | Tyre age |
|---|---|
| range | 14-33 laps (green flag 16-30; the two at 33 under the Safety Car) |
| lower quartile / median | 20 / 26 laps |

- **R5** blocks a pit on tyres younger than 10 laps on a dry track, unless the
  Safety Car, VSC or red flag makes the stop cheap. 10 leaves a margin under
  the earliest real stop (14); applied to the drivers' own decisions it blocks
  none of the 17 real stops.
- **R4 is removed.** A forced pit on old tyres has no support in the data: no
  stop happened above 33 laps, yet sets were run up to 50 laps old to the
  finish (17 laps above 33 without a stop), and the safe limit depends on
  compound and track. A threshold inside the data range would override real
  decisions, one above it would never fire. The id R4 stays free so R5-R10
  keep their numbers.
- **Reward:** pitting from age 20 (the pit window) or in rain/damp +1.5,
  pitting below age 10 on a dry track without SC/VSC/red -1.0, in between 0.
  The old penalty for staying out on "low tyre life" (i.e. on new tyres) and
  the `+0.3 * TyreLife` term in the base reward are gone.

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
python evaluate_holdout.py   # 3 seeds x 3 folds, ~1 h on a laptop CPU, writes docs/holdout_results.md
```

Training is not reproducible enough to trust a single run: with the same code
and data, compound accuracy on one held-out track moved by up to ~20 points
between training seeds. `evaluate_holdout.py` therefore trains every fold with
three seeds and reports the mean, the range and one row per seed
([`docs/holdout_results.md`](docs/holdout_results.md)).

Held-out results, SAFE = policy + safety layer, mean over seeds 0, 1, 2:

| Held-out track | Compound (min-max over seeds) | Baseline | Style | Baseline | Pit recall | Pit precision | Pit rate |
|---|---|---|---|---|---|---|---|
| Silverstone | 68.3 % (52-80 %) | 6.8 % | 45.9 % | 56.6 % | 33 % | 62 % | 1.3 % |
| Monza | 52.8 % (35-73 %) | 41.7 % | 32.5 % | 41.1 % | 67 % | 62 % | 2.1 % |
| Barcelona | 49.7 % (48-51 %) | 29.9 % | 46.6 % | 52.9 % | 92 % | 12 % | 22.3 % |
| **mean** | **56.9 % (46-67 %)** | 26.1 % | **41.6 %** | 50.2 % | **69 %** | 18 % | 8.6 % |

Per seed, the agent catches 14, 14 and 7 of the 17 real pit stops (11.7 on
average) with a pooled precision of 17-18 %.

- **Compound choice generalises:** above the baseline on average and on every
  track except Monza in one seed, but the spread between seeds is large.
- **Driving style does not generalise:** below the majority-class baseline on
  every track. With a label from measured pace instead of a rule over the
  agent's own inputs, the current features are not enough to predict it.
- **Pit stops:** most false pit calls come from Barcelona (22 % of laps); on the
  other two tracks the agent pits on ~2 % of laps, close to the real 2.5 %.
  Pit stops are rare, so accuracy is not a meaningful metric for them (a model
  that never pits reaches 97.5 %); recall and precision are reported instead.

**Effect of the tyre-age fix** (held-out, mean over seeds; before: three runs
of the previous code):

| | Before | After |
|---|---|---|
| Compound | 54.8 % (51-61 %) | 56.9 % (46-67 %) |
| Real pit stops caught (of 17) | 10.3 (14, 13, 4) | 11.7 (14, 14, 7) |
| Pit precision (pooled) | 14 % | 18 % |
| Pit rate | 11.0 % | 8.6 % |
| Real stops blocked by the safety layer | 2 in every run (old R5, age 33 under SC) | 0 |
| Style | 42.4 % | 41.6 % |

The compound and style differences are within the seed-to-seed spread. The
clear effects are on pits: the safety layer no longer blocks real stops, and
the agent makes fewer false pit calls.

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

### Previous numbers (historical)

An early version was evaluated only on the 674 laps it was trained on, with
random gaps and a gap-based style rule: compound 69.6 % (baseline 43.9 %),
style 64.2 % (baseline 42.1 %), pit recall 88 % / precision 24 %. The first
held-out run of the corrected data (one seed, before the tyre-age fix) gave
compound 52.8 % (baseline 26.1 %), style 43.2 %, 14 of 17 pit stops. These are
kept only to document how the evaluation changed.

---

## Prerequisites

- Python 3.10 or newer
- Internet access the first time `prepare_data.py` runs (FastF1 downloads and caches the data)

The trained models (`saved_models/`) and the prepared data (`data/`) are committed, so the API and the evaluation can run without retraining.

> **FastF1 account.** Since FastF1 3.9, downloading new sessions requires an
> F1TV Access (or higher) account; FastF1 asks for it on the first download.
> Sessions already in the local cache (`data/cache`) load without signing in.
> The Belgian GP was planned as the third track, but the Spanish GP
> (Barcelona) was already cached and is used instead.

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
python evaluate_holdout.py                # held-out: train on 2 tracks, test on the 3rd (all folds, seeds 0 1 2)
python evaluate_holdout.py --seeds 0      # one seed only (~20 min)
python evaluate_holdout.py --tracks Monza # one fold only
python train.py --holdout Monza           # train a single held-out model with plots
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
- **Pit frequency.** On held-out tracks the agent pits on ~9 % of laps on average (real drivers ~2.5 %), almost all of it at Barcelona; a consequence of the asymmetric imitation reward that prevents the opposite failure (never pitting).
- **Seed variance.** Results move by up to ~20 points between training seeds on a single track, so only averages over seeds are meaningful.
- **Checkpoint selection.** `train.py` keeps the checkpoint with the best reward per step, but the easy curriculum level has fewer synthetic rain/SC laps and scores higher, so an early checkpoint is usually selected. Evaluating every checkpoint at a fixed difficulty would remove this bias.
- **No tyre-age limit.** No rule forces a stop on very old tyres, because the data does not support a single age limit (see Tyre age above).
- **Not a physical race simulation.** The environment replays real laps with synthetic rain and Safety Car scenarios on top; `TyreLife` and other features do not change as a consequence of the agent's decisions.
- **No memory across laps.** `LSTMFeatureExtractor` in `model.py` is kept as a starting point for future work. Real memory over laps would require `RecurrentPPO` from `sb3-contrib` instead of standard PPO.

---

## Tech stack

Python · Stable-Baselines3 (PPO) · Gym · PyTorch · FastF1 · pandas · NumPy · scikit-learn · FastAPI · Pydantic · pytest · Matplotlib

## Author

Antonela Ković – [GitHub](https://github.com/antonela-kovic)

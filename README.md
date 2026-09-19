# F1 RL Agent – PPO Strategija

Ovaj projekt koristi PPO (Proximal Policy Optimization) za treniranje agenta koji optimizira strategiju utrkivanja (pit stop, izbor guma, stil vožnje) koristeći stvarne podatke iz FastF1 biblioteke.

---

## KORACI ZA POKRETANJE

### 1. Priprema i treniranje agenta

```bash
python prepare_data.py        # Generira podatke za treniranje (više vozača i trka)
python train.py               # Pokreće PPO treniranje s curriculum logikom
```

> Koriste se podaci iz trka: Silverstone, Monza, Spa (2023), vozači: VER, HAM, LEC, PER

---

### 2. Pokretanje REST API-ja

U korijenskom direktoriju pokreni FastAPI:

```bash
uvicorn api:app --reload
```

Alternativa:

```bash
python -m uvicorn api:app --reload
```

* API dostupan na: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Swagger sučelje: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 3. Evaluacija i testiranje

#### U VS Code

* Otvori `main.ipynb` i izvrši sve ćelije (`Shift + Enter`)
* Provjeri točnost agenta i ponašanje po tipu vremena

#### Alternativa (Jupyter)

```bash
jupyter notebook
```

Zatim otvori `main.ipynb`.

---

## Funkcionalnosti

*  Višestazni podaci i više vozača
*  Imitacija ljudskih odluka (PIT, compound, stil)
*  Vremenski uvjeti: suho, mokro, kiša
*  Curriculum težina (easy → medium → full)
*  REST API s objašnjenjima odluka
*  Vizualizacije:

  * Reward po epizodi (smoothed)
  * Agent vs Vozač ponašanje
  * Reward po vremenskim uvjetima
  * Distribucija guma

---

## Evaluacija agenta

U `main.ipynb` dostupni su grafovi i točnost predikcije:

* `Točnost izbora PIT-a`
* `Točnost izbora GUMA`
* `Točnost STILA vožnje`
* Interpretacija akcije (npr. "PIT: DA, Gume: Inter, Stil: Štedljivo")

---

## REST API endpointi

| Endpoint                       | Opis                                              |
| ------------------------------ | ------------------------------------------------- |
| `GET /`                        | Test da je server pokrenut                        |
| `POST /predict_strategy`       | Predikcija strategije za jedno stanje             |
| `POST /predict_strategy_batch` | Predikcija strategije za više stanja (lista dict) |
| `POST /explain_strategy`       | Objašnjenje strategije na temelju ulaznog stanja  |

> Napomena: za `batch`, očekuje se: `[{"state": [...]}, {"state": [...]}]`

Primjeri za testiranje u Swagger sučelju (`/docs`), za `POST /predict_strategy`:

```json
// 1. Suha staza, dobre gume
{"state": [20, 1, 12, 2, 1, 5, 91.2, 30.5, 30.1, 1.8, 2.2, 1, 30.0, 22.0, 1]}

// 2. Mokra staza, istrošene gume
{"state": [5, 2, 22, 2, 1, 8, 100.5, 35.0, 34.0, 0.5, 1.0, 2, 23.0, 18.0, 2]}

// 3. Kišni uvjeti, agresivan stil
{"state": [7, 3, 18, 1, 1, 4, 105.0, 36.5, 35.2, 0.8, 1.5, 1, 19.0, 16.0, 3]}

// 4. Vruća staza, medium compound
{"state": [15, 1, 9, 1, 1, 3, 95.0, 31.0, 30.0, 2.5, 3.0, 0, 36.0, 28.0, 1]}
```

---

## Testiranje API-ja

Pokreni `main.ipynb` za:

* Slanje zahtjeva prema API-ju
* Provjeru interpretacija i odgovora
* Prikaz grafova po koraku (reward, akcije)

---

## Ovisnosti

`requirements.txt` sadrži sve potrebno (uključujući `shimmy` i `gym`, potrebne jer projekt koristi stariju `gym` API-ju uz noviji `stable-baselines3`):

```bash
pip install -r requirements.txt
```

---

## Struktura projekta

```
VS_Code verzija/
├\api.py                   # FastAPI sučelje
├\f1_env.py                # Okruženje s nagradnom funkcijom i vremenskim uvjetima
├\model.py                 # LSTMFeatureExtractor - eksperimentalna komponenta za budući
│                           # rad, trenutno se NE koristi (train.py koristi standardnu
│                           # MLP arhitekturu 256-128-64 definiranu izravno u PPO pozivu)
├\train.py                 # PPO treniranje s evaluacijama
├\prepare_data.py          # Učitavanje i spremanje FastF1 podataka
├\.ipynb                   # Evaluacija + API testiranje
├\predict_strategies.py    # Evaluacija RAW vs SAFE (logički sloj), R10 provjera po utrci
├\utils.py                 # Helper funkcije, značajke, skaliranje opažanja
├\saved_models/            # Trenirani modeli
├\data/                    # .npy i CSV podaci
└\requirements.txt         # Python ovisnosti
```

---

## Testiranje

Projekt ima jedinične testove (pytest) za dijelove koda koji ne ovise o
FastF1 mreži ni o treniranom modelu - pravila u `logic_layer.py`, mehaniku
okruženja u `f1_env.py` (segmentacija utrka, miješanje redoslijeda, per-utrka
bookkeeping) i pomoćne funkcije u `utils.py`:

```bash
pytest -v
```

`prepare_data.py` se namjerno ne testira izravno jer na vrhu datoteke odmah
pokreće FastF1 mrežne pozive - njegove čiste (bez I/O) pomoćne funkcije
(`classify_weather`, `worst_track_status`) žive u `utils.py` upravo zato da
budu testabilne izolirano.

---

## Poznata ograničenja

* `FrontGap`/`RearGap` (razmak do vozila ispred/iza) su sintetički generirani - FastF1 ne pruža pouzdan izvor za tu značajku na razini kruga.
* `Style` (stil vožnje) ground truth je heuristički proxy izveden iz `FrontGap`/`RearGap`/`TyreLife`, ne stvarna telemetrija (u F1 podacima ne postoji izravna oznaka "stila").
* Agent trenutno pita nešto češće od stvarnih vozača (~9-10% naspram stvarnih ~2.5%) - posljedica asimetrične nagrade uvedene da spriječi suprotan problem (agent koji nikad ne pita). Logički sloj (`logic_layer.py`) i dalje osigurava usklađenost sa strateškim pravilima neovisno o ovome.
* Okruženje (`f1_env.py`) simulira raznolike scenarije (kišu, Safety Car) preko postojećih stvarnih podataka; nije puna fizička simulacija utrke - `TyreLife` i ostale značajke ne mijenjaju se dinamički kao posljedica agentovih odluka unutar jednog prolaska kroz podatke.
* `LSTMFeatureExtractor` (`model.py`) je pripremljen kao osnova za budući rad prema recurrentnoj politici, ali trenutno nije uključen u trening - za stvarnu memoriju kroz krugove bio bi potreban `RecurrentPPO` (sb3-contrib), ne standardni `PPO`.

---


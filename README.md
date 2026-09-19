
### F1 RL Agent – PPO Strategija + Logički sloj (RAW → SAFE)

Ovaj projekt koristi **PPO (Proximal Policy Optimization)** za treniranje agenta koji optimizira strategiju utrkivanja (pit stop, izbor guma, stil vožnje) koristeći stvarne podatke iz **FastF1** biblioteke. Uveden je **deterministički logički sloj post-verifikacije** koji provjerava PPO odluke i po potrebi radi korekciju:
- **RAW** = originalna PPO odluka
- **SAFE** = odluka nakon provjere pravila + korekcija (override)
- generira se i **audit report** (kršenja + što je promijenjeno i zašto)

---

### KORACI ZA POKRETANJE

### 0) Instalacija ovisnosti

```bash
pip install -r requirements.txt
````

Ako koristiš API:

```bash
pip install fastapi uvicorn
```

> Napomena: projekt koristi Stable-Baselines3 (PPO), Gymnasium/Gym kompatibilnost, FastF1 i standardne znanstvene biblioteke (NumPy, Pandas, Matplotlib…).

---

### 1) Priprema podataka (FastF1 → CSV/NPY)

```bash
python prepare_data.py
python convert_to_npy.py
```

* `prepare_data.py` dohvaća i priprema podatke (više vozača i više utrka)
* `convert_to_npy.py` generira NPY datoteke koje se koriste u treningu/evaluaciji

Primjeri korištenih podataka (ovisno o konfiguraciji skripti):

* staze: Silverstone, Monza, Spa (npr. 2023)
* vozači: VER, HAM, LEC, PER

---

### 2) Treniranje PPO agenta

```bash
python train.py
```

Trening uključuje:

* PPO agent (policy + value mreža)
* curriculum logiku (easy → medium → full) ovisno o implementaciji u `train.py`
* spremanje modela u `saved_models/`

---

### 3) Brzi test okruženja (opcionalno)

Ako želiš brzo provjeriti da env radi i da NPY datoteke postoje:

```bash
python test_env.py
```

---

### 4) Evaluacija / Analiza rezultata

Možeš koristiti:

* `main.ipynb` (notebook evaluacija + grafovi)
* `main.py` (ako ti je lakše skriptno)

---

## LOGIČKI SLOJ (post-verification)

Logički sloj je implementiran u:

* `logic_layer.py`

Radi kao **post-processing** nad PPO odlukom:

* ulaz: `(state, raw_action)`
* izlaz: `(safe_action, report)`

### Što report sadrži?

* `violations_before`: popis detektiranih kršenja pravila na RAW akciji
* `overrides`: popis korekcija (koje polje je promijenjeno, RAW → SAFE, i objašnjenje)

### Tipična pravila (primjer ideje)

* ako je `WET` → dopuštene samo `INTER/WET` gume
* ako je `TyreLife` ispod praga → zabrani agresivni stil
* ako je `FrontGap` mali → ublaži stil (aggressive → normal/conservative)

> Pravila su namjerno “minimalna obrana” (hard constraints) i cilj im je smanjiti nelogične ili domenski neispravne odluke.

---

## REST API (FastAPI)

Pokretanje:

```bash
uvicorn api:app --reload
```

* API: `http://127.0.0.1:8000`
* Swagger: `http://127.0.0.1:8000/docs`

### Endpointi

| Endpoint                       | Opis                                                                   |
| ------------------------------ | ---------------------------------------------------------------------- |
| `GET /`                        | Health-check                                                           |
| `POST /predict_strategy`       | Predikcija strategije za jedno stanje (**RAW + SAFE + report**)        |
| `POST /predict_strategy_batch` | Predikcija za više stanja + agregirane statistike (npr. override rate) |
| `POST /explain_strategy`       | Objašnjenje odluke (interpretacija) + logički report                   |

**Ulazni format (primjer):**

```json
{
  "state": [ ... ]
}
```

**Batch format:**

```json
[
  {"state": [ ... ]},
  {"state": [ ... ]}
]
```

---

## Funkcionalnosti (sažetak)

* Višestazni podaci i više vozača (FastF1)
* PPO agent treniran za strategijske odluke:

  * PIT stop
  * tyre compound
  * stil vožnje
* Vremenski uvjeti uključeni u stanje (kao feature)
* Curriculum learning (ovisno o `train.py`)
* Logički sloj post-verifikacije: RAW → SAFE + audit report
* REST API za integraciju i online testiranje
* Evaluacija i vizualizacije (reward, distribucije, usporedbe)

---

## Ulazne značajke (state)

Odabrane značajke su definirane u `utils.py` kao `selected_features` (primjer):

* `TyreLife`, `CompoundEncoded`, `LapNumber`, `TrackStatus`, `Position`, `LapTime`, `Sector1Time`, `Sector2Time`,
* `FrontGap`, `RearGap`, `Stint`, `TrackTemperature`, `AirTemperature`,
* `Weather`, …

> Dimenzionalnost stanja ovisi o listi `selected_features` i treba biti konzistentna između prepare_data → env → model.

---

## Struktura projekta

```
VS_Code verzija-finall/
├─ api.py                  # FastAPI sučelje (RAW + SAFE + report)
├─ logic_layer.py          # Post-verifikacija pravila (override + audit)
├─ f1_env.py               # RL okruženje (reward, step, reset...)
├─ train.py                # PPO treniranje
├─ test_env.py             # Brzi test okruženja i podataka
├─ prepare_data.py         # Dohvat i priprema FastF1 podataka
├─ convert_to_npy.py       # Konverzija u .npy (obs/actions)
├─ finetune.py             # (opcionalno) dodatno ugađanje
├─ main.py                 # (opcionalno) skriptna evaluacija
├─ main.ipynb              # evaluacija + grafovi + API testiranje
├─ utils.py                # pomoćne funkcije + selected_features
├─ data/                   # .npy / CSV podaci
├─ saved_models/           # spremljeni modeli
└─ requirements.txt        # ovisnosti
```

---

## Tipični workflow (najkraće)

1. `python prepare_data.py`
2. `python convert_to_npy.py`
3. `python train.py`
4. (opcionalno) `python test_env.py`
5. `uvicorn api:app --reload` + test preko `/docs`
6. evaluacija u `main.ipynb`



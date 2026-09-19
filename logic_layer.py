# logic_layer.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np

# Compound mapping (iz tvog projekta)
# 0 SOFT, 1 MEDIUM, 2 HARD, 3 INTER, 4 WET
COMPOUND_NAMES = {0: "SOFT", 1: "MEDIUM", 2: "HARD", 3: "INTER", 4: "WET"}
STYLE_NAMES = {0: "Konzervativno", 1: "Neutralno", 2: "Agresivno"}

@dataclass
class LogicReport:
    violations_before: List[str]
    overrides: List[Dict[str, Any]]
    rules_triggered: List[str]

def _idx(selected_features: List[str], name: str) -> int:
    return selected_features.index(name)

def verify_action(
    state: np.ndarray,
    action: np.ndarray,
    selected_features: List[str],
    config: Dict[str, float] | None = None
) -> Tuple[np.ndarray, LogicReport]:
    """
    state: shape (n_features,)
    action: shape (3,) -> [pit, compound, style]
    """
    if config is None:
        config = {
            "tyre_pit_force": 2.0,     # samo kad je baš kritično
            "tyre_pit_block": 30.0,    # blokiraj pit samo kad su gume jako dobre
        }

    # ensure types
    s = np.asarray(state, dtype=np.float32).copy()
    a = np.asarray(action, dtype=np.int32).copy()
    pit, comp, style = int(a[0]), int(a[1]), int(a[2])

    # extract facts
    i_weather = _idx(selected_features, "Weather")
    i_tyre = _idx(selected_features, "TyreLife")

    weather = int(round(float(s[i_weather])))
    tyre_life = float(s[i_tyre])

    # facts -> predicates
    wet = weather == 3
    damp = weather == 2
    dry = weather == 1

    violations = []
    overrides = []
    triggered = []

    def override(field: str, old: int, new: int, rule: str, why: str):
        overrides.append({"field": field, "from": old, "to": new, "rule": rule, "why": why})

    # ==== RULES (minimalni, obrambeni set) ====

    # R1: Wet => WET tyres
    if wet:
        triggered.append("R1_WET_REQUIRES_WET")
        if comp != 4:
            violations.append("R1: Weather=kiša -> compound mora biti WET(4)")
            override("compound", comp, 4, "R1_WET_REQUIRES_WET", "Kiša: WET je obavezno")
            comp = 4

  
    # R2: Damp => INTER tyres (enforced)
    if damp:
        triggered.append("R2_DAMP_REQUIRES_INTER")
        if comp != 3:
            violations.append("R2: Weather=mokro -> compound mora biti INTER(3)")
            override("compound", comp, 3, "R2_DAMP_REQUIRES_INTER", "Mokro: INTER je obavezno")
            comp = 3

    # R3: Dry => forbid INTER/WET
    i_temp = _idx(selected_features, "TrackTemperature")
    track_temp = float(s[i_temp])

    if dry:
        triggered.append("R3_DRY_RULES")
        # forbid INTER/WET
        if comp in (3, 4):
            violations.append("R3a: Suho -> INTER/WET nisu dozvoljeni")
            override("compound", comp, 1, "R3a_DRY_FORBIDS_INTER_WET", "Suho: prebacujem na MEDIUM")
            comp = 1

        # optional: SOFT on cool track, else MEDIUM
        target = 0 if track_temp < 20.0 else 1
        if comp not in (0, 1, 2):
            # already handled
            pass
        else:
            # if model chooses HARD too often, softly steer it
            if comp == 2:
                violations.append("R3b: Suho -> HARD rijetko optimalan u ovom datasetu (prefer MEDIUM/SOFT)")
                override("compound", comp, target, "R3b_DRY_PREFERS_SOFT_MED", f"TrackTemp={track_temp:.1f}")
                comp = target

    # R4: Very low tyre life => force PIT
    triggered.append("R4_LOW_TYRE_FORCES_PIT")
    if tyre_life < config["tyre_pit_force"]:
        if pit != 1:
            violations.append("R4: TyreLife jako nizak -> PIT mora biti 1")
            override("pit", pit, 1, "R4_LOW_TYRE_FORCES_PIT", f"TyreLife={tyre_life:.1f} < {config['tyre_pit_force']}")
            pit = 1

    # R5: Block pointless pit on dry + good tyres
    triggered.append("R5_BLOCK_POINTLESS_PIT")
    if dry and tyre_life > config["tyre_pit_block"]:
        if pit != 0:
            violations.append("R5: Suho + dobre gume -> PIT treba biti 0")
            override("pit", pit, 0, "R5_BLOCK_POINTLESS_PIT", f"TyreLife={tyre_life:.1f} > {config['tyre_pit_block']} i suho")
            pit = 0

 
    # R6: Wet/damp -> avoid aggressive style
    triggered.append("R6_WET_AVOIDS_AGGRESSIVE")
    if (wet or damp) and style == 2:
        violations.append("R6: Mokro/kiša -> agresivan stil nije dopušten (sigurnost)")
        override("style", style, 1, "R6_WET_AVOIDS_AGGRESSIVE", "Smanjujem rizik na mokrom")
        style = 1

    # Front gap (treba biti definiran neovisno o R6)
    i_front = _idx(selected_features, "FrontGap")
    front_gap = float(s[i_front])

    # R7: small gap -> conservative
    triggered.append("R7_SMALL_GAP_CONSERVATIVE")
    if front_gap < 1.5 and style != 0:
        violations.append("R7: Mali front gap -> preferiraj konzervativno")
        override("style", style, 0, "R7_SMALL_GAP_CONSERVATIVE", f"FrontGap={front_gap:.2f} < 1.5")
        style = 0

    # R8: big gap + dry -> allow aggressive (soft steer)
    triggered.append("R8_BIG_GAP_DRY_MORE_AGGRESSIVE")
    if dry and front_gap > 2.5 and style == 0:
        override("style", style, 1, "R8_BIG_GAP_DRY_MORE_AGGRESSIVE", f"FrontGap={front_gap:.2f} > 2.5")
        style = 1

    safe_action = np.array([pit, comp, style], dtype=np.int32)

    report = LogicReport(
        violations_before=violations,
        overrides=overrides,
        rules_triggered=triggered
    )
    return safe_action, report

def humanize_action(action: np.ndarray) -> Dict[str, str]:
    pit, comp, style = int(action[0]), int(action[1]), int(action[2])
    return {
        "pit": "DA" if pit == 1 else "NE",
        "compound": f"{comp} ({COMPOUND_NAMES.get(comp, 'UNKNOWN')})",
        "style": f"{style} ({STYLE_NAMES.get(style, 'UNKNOWN')})",

    }

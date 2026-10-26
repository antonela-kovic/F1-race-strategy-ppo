# safety_layer.py
# Rule-based safety layer: verifies every action proposed by the PPO policy
# against explicit race-strategy rules (R1-R10) and overrides it when a rule
# is violated. Every override records the rule and the reason.
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np

# Compound mapping
# 0 SOFT, 1 MEDIUM, 2 HARD, 3 INTER, 4 WET
COMPOUND_NAMES = {0: "SOFT", 1: "MEDIUM", 2: "HARD", 3: "INTER", 4: "WET"}
STYLE_NAMES = {0: "Conservative", 1: "Neutral", 2: "Aggressive"}
DRY_COMPOUNDS = (0, 1, 2)  # SOFT, MEDIUM, HARD

# FastF1 TrackStatus codes: 1=clear, 2=yellow, 3=unused, 4=Safety Car,
# 5=red, 6=VSC, 7=VSC ending
CAUTION_STATUSES = {2, 4, 5, 6, 7}


@dataclass
class RaceState:
    """
    Keeps context for the WHOLE race/episode (not only the current lap),
    required by rules that cannot be checked from a single state row
    (e.g. "two dry compounds are mandatory in a dry race").

    Usage: after every lap, once the SAFE action is known, call
    `race_state.record(weather, safe_compound)` to update the history.
    """
    total_laps: int
    lap_index: int = 0
    compounds_used: Set[int] = field(default_factory=set)
    wet_or_damp_seen: bool = False
    force_window: int = 3  # how many laps before the end the compound change is forced

    @property
    def laps_remaining(self) -> int:
        return max(self.total_laps - self.lap_index - 1, 0)

    @property
    def is_dry_race_so_far(self) -> bool:
        # The two-compound rule does not apply if the race (or part of it) was declared wet
        return not self.wet_or_damp_seen

    @property
    def dry_compounds_used(self) -> Set[int]:
        return self.compounds_used & set(DRY_COMPOUNDS)

    def record(self, weather: int, compound: int) -> None:
        if weather in (2, 3):
            self.wet_or_damp_seen = True
        self.compounds_used.add(compound)
        self.lap_index += 1


@dataclass
class SafetyReport:
    violations_before: List[str]
    overrides: List[Dict[str, Any]]
    rules_triggered: List[str]

def _idx(selected_features: List[str], name: str) -> int:
    return selected_features.index(name)

def verify_action(
    state: np.ndarray,
    action: np.ndarray,
    selected_features: List[str],
    config: Dict[str, float] | None = None,
    race_state: Optional[RaceState] = None,
) -> Tuple[np.ndarray, SafetyReport]:
    """
    state: shape (n_features,)
    action: shape (3,) -> [pit, compound, style]
    race_state: optional whole-race context (tyre history, laps remaining).
        Without it, R9 (Safety Car/VSC) still works because it relies only on
        the current state, but R10 (two mandatory dry compounds) is skipped
        because it cannot be checked without the history of previous laps.
    """
    if config is None:
        config = {
            # Thresholds were relaxed (from 5.0 / 15.0) because stricter values
            # overrode too many correct pit decisions and reduced pit accuracy.
            "tyre_pit_force": 2.0,     # force a pit only when tyre life is critical
            "tyre_pit_block": 30.0,    # block a pit only when the tyres are very good
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

    # ==== RULES (minimal, defensive set) ====

    # R1: Wet => WET tyres
    if wet:
        triggered.append("R1_WET_REQUIRES_WET")
        if comp != 4:
            violations.append("R1: Weather=rain -> compound must be WET(4)")
            override("compound", comp, 4, "R1_WET_REQUIRES_WET", "Rain: WET is mandatory")
            comp = 4

    # R2: Damp => INTER tyres (enforced)
    if damp:
        triggered.append("R2_DAMP_REQUIRES_INTER")
        if comp != 3:
            violations.append("R2: Weather=damp -> compound must be INTER(3)")
            override("compound", comp, 3, "R2_DAMP_REQUIRES_INTER", "Damp: INTER is mandatory")
            comp = 3

    # R3: Dry => forbid INTER/WET
    if dry:
        triggered.append("R3_DRY_RULES")
        if comp in (3, 4):
            violations.append("R3a: Dry -> INTER/WET are not allowed")
            override("compound", comp, 1, "R3a_DRY_FORBIDS_INTER_WET", "Dry: switching to MEDIUM")
            comp = 1
        # Note: there is intentionally no rule that rewrites HARD choices on a
        # dry track. It is not an FIA rule, and HARD is used as often as
        # MEDIUM/SOFT in the real data, so such a rule would erase a correctly
        # learned signal.

    # R4: Very low tyre life => force PIT
    triggered.append("R4_LOW_TYRE_FORCES_PIT")
    if tyre_life < config["tyre_pit_force"]:
        if pit != 1:
            violations.append("R4: TyreLife very low -> PIT must be 1")
            override("pit", pit, 1, "R4_LOW_TYRE_FORCES_PIT", f"TyreLife={tyre_life:.1f} < {config['tyre_pit_force']}")
            pit = 1

    # R5: Block pointless pit on dry + good tyres
    triggered.append("R5_BLOCK_POINTLESS_PIT")
    if dry and tyre_life > config["tyre_pit_block"]:
        if pit != 0:
            violations.append("R5: Dry + good tyres -> PIT should be 0")
            override("pit", pit, 0, "R5_BLOCK_POINTLESS_PIT", f"TyreLife={tyre_life:.1f} > {config['tyre_pit_block']} and dry")
            pit = 0

    # R6: Wet/damp -> avoid aggressive style
    triggered.append("R6_WET_AVOIDS_AGGRESSIVE")
    if (wet or damp) and style == 2:
        violations.append("R6: Damp/rain -> aggressive style is not allowed (safety)")
        override("style", style, 1, "R6_WET_AVOIDS_AGGRESSIVE", "Reducing risk on a wet track")
        style = 1

    # Front gap (must be defined independently of R6)
    i_front = _idx(selected_features, "FrontGap")
    front_gap = float(s[i_front])

    # R7: small gap -> conservative
    triggered.append("R7_SMALL_GAP_CONSERVATIVE")
    if front_gap < 1.5 and style != 0:
        violations.append("R7: Small front gap -> prefer conservative")
        override("style", style, 0, "R7_SMALL_GAP_CONSERVATIVE", f"FrontGap={front_gap:.2f} < 1.5")
        style = 0

    # R8: big gap + dry -> allow aggressive (soft steer)
    triggered.append("R8_BIG_GAP_DRY_MORE_AGGRESSIVE")
    if dry and front_gap > 2.5 and style == 0:
        override("style", style, 1, "R8_BIG_GAP_DRY_MORE_AGGRESSIVE", f"FrontGap={front_gap:.2f} > 2.5")
        style = 1

    # R9: Safety Car / VSC / red / yellow -> overtaking is forbidden by the
    # regulations, so an aggressive style (which only makes sense when trying
    # to overtake) is not allowed.
    if "TrackStatus" in selected_features:
        i_track = _idx(selected_features, "TrackStatus")
        track_status = int(round(float(s[i_track])))
        caution = track_status in CAUTION_STATUSES

        triggered.append("R9_CAUTION_FORBIDS_AGGRESSIVE")
        if caution and style == 2:
            violations.append(
                f"R9: TrackStatus={track_status} (SC/VSC/red/yellow) -> no overtaking, aggressive style is not allowed"
            )
            override("style", style, 0, "R9_CAUTION_FORBIDS_AGGRESSIVE",
                      f"TrackStatus={track_status}: no overtaking under caution")
            style = 0

    # R10: At least two different DRY compounds must be used during a dry race
    # (FIA Sporting Regulations). The rule does not apply if the race was ever
    # wet/damp (it is then a "wet race").
    # This rule CANNOT be checked from a single lap and requires race_state.
    if race_state is not None:
        triggered.append("R10_MANDATORY_TWO_DRY_COMPOUNDS")

        # compound that would be applied AFTER all previous overrides
        prospective_dry_used = race_state.dry_compounds_used | ({comp} if comp in DRY_COMPOUNDS else set())

        if (
            race_state.is_dry_race_so_far
            and dry  # the current lap is still dry
            and len(prospective_dry_used) < 2
            and race_state.laps_remaining <= race_state.force_window
        ):
            # force a pit stop if the agent would not make one
            if pit != 1:
                violations.append(
                    "R10a: Only one dry compound used so far near the end of the race -> PIT must be 1"
                )
                override("pit", pit, 1, "R10_MANDATORY_TWO_DRY_COMPOUNDS",
                          f"LapsRemaining={race_state.laps_remaining} <= {race_state.force_window}, "
                          f"dry_compounds_used={sorted(race_state.dry_compounds_used)}")
                pit = 1

            # force a DIFFERENT dry compound (not the one already used)
            already_used = race_state.dry_compounds_used
            if comp in already_used or not already_used:
                candidates = [c for c in DRY_COMPOUNDS if c not in already_used]
                new_comp = candidates[0] if candidates else comp
                if new_comp != comp:
                    violations.append(
                        f"R10b: Compound={COMPOUND_NAMES.get(comp)} has been used for the whole race -> "
                        f"a change to another dry compound is mandatory"
                    )
                    override("compound", comp, new_comp, "R10_MANDATORY_TWO_DRY_COMPOUNDS",
                              f"Used so far: {sorted(already_used) if already_used else 'none (first pit)'}")
                    comp = new_comp

    safe_action = np.array([pit, comp, style], dtype=np.int32)

    report = SafetyReport(
        violations_before=violations,
        overrides=overrides,
        rules_triggered=triggered
    )
    return safe_action, report

def humanize_action(action: np.ndarray) -> Dict[str, str]:
    pit, comp, style = int(action[0]), int(action[1]), int(action[2])
    return {
        "pit": "YES" if pit == 1 else "NO",
        "compound": f"{comp} ({COMPOUND_NAMES.get(comp, 'UNKNOWN')})",
        "style": f"{style} ({STYLE_NAMES.get(style, 'UNKNOWN')})",
    }

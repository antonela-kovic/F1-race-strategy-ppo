# safety_layer.py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np

# Compound mapping (iz tvog projekta)
# 0 SOFT, 1 MEDIUM, 2 HARD, 3 INTER, 4 WET
COMPOUND_NAMES = {0: "SOFT", 1: "MEDIUM", 2: "HARD", 3: "INTER", 4: "WET"}
STYLE_NAMES = {0: "Konzervativno", 1: "Neutralno", 2: "Agresivno"}
DRY_COMPOUNDS = (0, 1, 2)  # SOFT, MEDIUM, HARD

# FastF1 TrackStatus kodovi: 1=čisto, 2=žuti, 3=nekorišteno, 4=Safety Car,
# 5=crveni, 6=VSC, 7=kraj VSC-a
CAUTION_STATUSES = {2, 4, 5, 6, 7}


@dataclass
class RaceState:
    """
    Pamti kontekst na razini CIJELE utrke/epizode (a ne samo trenutnog kruga),
    potrebno za pravila koja se ne mogu provjeriti iz jednog reda stanja
    (npr. "obavezna 2 tipa suhih guma po suhoj utrci").

    Poziv: nakon svakog kruga, kad je SAFE akcija poznata, pozovi
    `race_state.record(weather, safe_compound)` da se povijest ažurira.
    """
    total_laps: int
    lap_index: int = 0
    compounds_used: Set[int] = field(default_factory=set)
    wet_or_damp_seen: bool = False
    force_window: int = 3  # koliko krugova prije kraja prisiljavamo promjenu

    @property
    def laps_remaining(self) -> int:
        return max(self.total_laps - self.lap_index - 1, 0)

    @property
    def is_dry_race_so_far(self) -> bool:
        # Pravilo o 2 compounda ne vrijedi ako je utrka (ili dio nje) proglašena mokrom
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
    config: Dict[str, float] | None = None,
    race_state: Optional[RaceState] = None,
) -> Tuple[np.ndarray, LogicReport]:
    """
    state: shape (n_features,)
    action: shape (3,) -> [pit, compound, style]
    race_state: opcionalno - kontekst cijele utrke (povijest guma, broj krugova
        do kraja). Bez njega, R9 (safety car/VSC) i dalje rade jer se oslanjaju
        samo na trenutni state, ali R10 (obavezna 2 compounda) se preskače jer
        je po definiciji nemoguće provjeriti bez povijesti prijašnjih krugova.
    """
    if config is None:
        config = {
            # "tyre_pit_force": 5.0,   # ispod ove vrijednosti forsiraj pit
            # "tyre_pit_block": 15.0,  # iznad ove vrijednosti blokiraj pit u suhim uvjetima
            # Izmjena zbog narušavanja tocnosti pit stopa
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

    # R2: Damp => INTER preferred (soft rule but we enforce in prototype)
    # R2: Damp => INTER tyres (enforced)
    if damp:
        triggered.append("R2_DAMP_REQUIRES_INTER")
        if comp != 3:
            violations.append("R2: Weather=mokro -> compound mora biti INTER(3)")
            override("compound", comp, 3, "R2_DAMP_REQUIRES_INTER", "Mokro: INTER je obavezno")
            comp = 3

    # R3: Dry => forbid INTER/WET
    if dry:
        triggered.append("R3_DRY_RULES")
        # forbid INTER/WET
        if comp in (3, 4):
            violations.append("R3a: Suho -> INTER/WET nisu dozvoljeni")
            override("compound", comp, 1, "R3a_DRY_FORBIDS_INTER_WET", "Suho: prebacujem na MEDIUM")
            comp = 1

        # NAPOMENA: ovdje je ranije postojalo R3b - "meko" prepisivanje SVAKOG
        # HARD(2) izbora u SOFT/MEDIUM na temelju praga temperature staze.
        # Uklonjeno je jer NIJE stvarno FIA pravilo (za razliku od R3a) - bila
        # je to ad-hoc heuristika iz faze kad je RAW politika bila zaglavljena
        # i rijetko birala HARD. Otkad RAW politika stvarno uči birati HARD u
        # skladu sa stvarnom vozačevom strategijom (HARD se koristi 223 puta u
        # stvarnim podacima - usporedivo s MEDIUM/SOFT), ovo pravilo je
        # aktivno brisalo taj naučeni signal i vraćalo SAFE izlaz na lošiju,
        # manje točnu procjenu nego sam RAW izlaz.

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

    # R9: Safety Car / VSC / crveni / žuti -> pretjecanje je zabranjeno pravilnikom,
    # pa agresivan stil (koji ima smisla samo uz pokušaj pretjecanja) nije dopušten.
    if "TrackStatus" in selected_features:
        i_track = _idx(selected_features, "TrackStatus")
        track_status = int(round(float(s[i_track])))
        caution = track_status in CAUTION_STATUSES

        triggered.append("R9_CAUTION_FORBIDS_AGGRESSIVE")
        if caution and style == 2:
            violations.append(
                f"R9: TrackStatus={track_status} (SC/VSC/crveni/žuti) -> pretjecanje zabranjeno, agresivan stil nije dopušten"
            )
            override("style", style, 0, "R9_CAUTION_FORBIDS_AGGRESSIVE",
                      f"TrackStatus={track_status}: nema pretjecanja pod caution uvjetima")
            style = 0

    # R10: Obavezna upotreba barem 2 različita tipa SUHIH guma tijekom suhe utrke
    # (FIA Sporting Regulations, čl. 30/33). Pravilo se ne primjenjuje ako je
    # tijekom utrke ikad bilo mokro/vlažno (tada je utrka "wet race").
    # Ovo pravilo se NE MOŽE provjeriti iz jednog kruga - zahtijeva race_state.
    if race_state is not None:
        triggered.append("R10_MANDATORY_TWO_DRY_COMPOUNDS")

        # compound koji bi bio primijenjen NAKON svih dosadašnjih override-a
        prospective_dry_used = race_state.dry_compounds_used | ({comp} if comp in DRY_COMPOUNDS else set())

        if (
            race_state.is_dry_race_so_far
            and dry  # trenutni krug je i dalje suh
            and len(prospective_dry_used) < 2
            and race_state.laps_remaining <= race_state.force_window
        ):
            # prisili pit stop ako ga agent ne bi napravio
            if pit != 1:
                violations.append(
                    "R10a: Do kraja utrke korišten je samo 1 tip suhih guma -> PIT mora biti 1"
                )
                override("pit", pit, 1, "R10_MANDATORY_TWO_DRY_COMPOUNDS",
                          f"LapsRemaining={race_state.laps_remaining} <= {race_state.force_window}, "
                          f"dry_compounds_used={sorted(race_state.dry_compounds_used)}")
                pit = 1

            # prisili odabir DRUGOG tipa suhe gume (ne onog koji je već korišten)
            already_used = race_state.dry_compounds_used
            if comp in already_used or not already_used:
                candidates = [c for c in DRY_COMPOUNDS if c not in already_used]
                new_comp = candidates[0] if candidates else comp
                if new_comp != comp:
                    violations.append(
                        f"R10b: Compound={COMPOUND_NAMES.get(comp)} je već korišten cijelu utrku -> "
                        f"obavezna promjena na drugi suhi tip"
                    )
                    override("compound", comp, new_comp, "R10_MANDATORY_TWO_DRY_COMPOUNDS",
                              f"Dosad korišteno: {sorted(already_used) if already_used else 'ništa (prvi pit)'}")
                    comp = new_comp

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

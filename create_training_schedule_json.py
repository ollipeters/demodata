import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Match:
    match_id: str
    match_date: date
    matchday: Optional[int]
    opponent: Optional[str]
    home_away: Optional[str]
    total_distance_gross_km: Optional[float]


LABEL_TO_FACTOR: Dict[str, float] = {
    "MD": 1.0,
    "MD-1": 0.18,
    "MD-2": 0.40,
    "MD-3": 0.65,
    "MD-4": 0.35,
    "MD+1": 0.125,  # 10–15% as a single default
    "MD+2": 0.0,
}

LABEL_TO_SESSION_NAME: Dict[str, str] = {
    "MD": "Match",
    "MD-1": "Finishing",
    "MD-2": "Training",
    "MD-3": "Main Load",
    "MD-4": "Training",
    "MD+1": "Regeneration",
    "MD+2": "Off",
}


TACTICAL_CONTENTS: List[str] = [
    "Übungsform Spiel durchs Zentrum",
    "Übungsform Spielverlagerung",
    "Übungsform Aufbau unter Druck",
    "Übungsform Gegenpressing",
]

ATHLETIC_CYCLE: List[str] = [
    "Beschleunigungsläufe",
    "Beschleunigungsläufe",
    "RSA",
    "RSA",
    "MaxSpeed",
    "MaxSpeed",
]


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def normalize_team_name(value: str) -> str:
    return value.strip()


def load_matches_for_team(team_matches_json: Path, team_name: str) -> List[Match]:
    rows = load_json(team_matches_json)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {team_matches_json}, got {type(rows)}")

    matches: List[Match] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if normalize_team_name(str(row.get("team", ""))) != team_name:
            continue
        md = row.get("match_date")
        if not md:
            continue
        matches.append(
            Match(
                match_id=str(row.get("match_id", "")),
                match_date=parse_iso_date(str(md)),
                matchday=int(row["matchday"]) if row.get("matchday") is not None else None,
                opponent=row.get("opponent"),
                home_away=row.get("home_away"),
                total_distance_gross_km=safe_float(row.get("total_distance_gross_km")),
            )
        )

    matches = sorted(matches, key=lambda m: (m.match_date, m.match_id))
    return matches


def label_for_match_day_offset(offset_days: int) -> Optional[str]:
    if offset_days == 0:
        return "MD"
    if offset_days < 0:
        d = abs(offset_days)
        if d in (1, 2, 3, 4):
            return f"MD-{d}"
        return None
    if offset_days in (1, 2):
        return f"MD+{offset_days}"
    return None


def candidate_labels_for_date(d: date, matches: List[Match]) -> List[Tuple[int, str, Match]]:
    out: List[Tuple[int, str, Match]] = []
    for m in matches:
        offset = (d - m.match_date).days
        label = label_for_match_day_offset(offset)
        if label is None:
            continue
        out.append((offset, label, m))
    return out


def choose_best_candidate(candidates: List[Tuple[int, str, Match]]) -> Optional[Tuple[int, str, Match]]:
    if not candidates:
        return None

    # Prefer: closest in absolute days; in ties, prefer pre-match (negative offset)
    # because MD-2 should override MD+2 if a match is exactly 2 days away.
    def sort_key(item: Tuple[int, str, Match]) -> Tuple[int, int]:
        offset, _label, _match = item
        return (abs(offset), 0 if offset < 0 else 1)

    return sorted(candidates, key=sort_key)[0]


def daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def build_schedule(matches: List[Match]) -> List[Dict[str, Any]]:
    if not matches:
        return []

    start = matches[0].match_date - timedelta(days=4)
    end = matches[-1].match_date + timedelta(days=2)

    schedule: List[Dict[str, Any]] = []
    activation_counter = 0
    md3_ath_counter = 0
    for d in daterange(start, end):
        best = choose_best_candidate(candidate_labels_for_date(d, matches))
        if best is None:
            continue

        offset, label, m = best
        factor = LABEL_TO_FACTOR[label]
        base = m.total_distance_gross_km
        planned = (base * factor) if base is not None else None

        activation_counter += 1
        rondo_no = (activation_counter - 1) % 5 + 1
        tactical = TACTICAL_CONTENTS[(activation_counter - 1) % len(TACTICAL_CONTENTS)]
        athletic = None

        contents: List[Dict[str, Any]] = []
        # Always start with activation (numbered)
        contents.append({"type": "activation", "name": f"Aktivierung {activation_counter}"})

        # Build 2–5 contents depending on load day
        if label == "MD":
            contents.append({"type": "match", "name": "Match"})
        elif label == "MD+1":
            contents.append({"type": "regeneration", "name": "Regeneration"})
        elif label == "MD+2":
            contents.append({"type": "off", "name": "Off"})
        elif label == "MD-1":
            # Low load: 3 blocks
            contents.append({"type": "rondo", "name": f"Rondo {rondo_no}"})
            contents.append({"type": "tactical", "name": tactical})
        elif label == "MD-4":
            # Moderate/low: 3 blocks
            contents.append({"type": "rondo", "name": f"Rondo {rondo_no}"})
            contents.append({"type": "tactical", "name": tactical})
        elif label == "MD-2":
            # Medium: 4 blocks
            game = 6 + ((activation_counter - 1) % 2)  # 6v6 or 7v7
            contents.append({"type": "rondo", "name": f"Rondo {rondo_no}"})
            contents.append({"type": "tactical", "name": tactical})
            contents.append({"type": "game", "name": f"Spiel {game}v{game}"})
        elif label == "MD-3":
            # Main load: 5 blocks incl. athletic
            athletic = ATHLETIC_CYCLE[md3_ath_counter % len(ATHLETIC_CYCLE)]
            md3_ath_counter += 1
            game = 8 + ((activation_counter - 1) % 2)  # 8v8 or 9v9
            contents.append({"type": "athletic", "name": athletic})
            contents.append({"type": "rondo", "name": f"Rondo {rondo_no}"})
            contents.append({"type": "tactical", "name": tactical})
            contents.append({"type": "game", "name": f"Spiel {game}v{game}"})
        else:
            # Fallback: keep it minimal
            contents.append({"type": "rondo", "name": f"Rondo {rondo_no}"})

        schedule.append(
            {
                "date": d.isoformat(),
                "label": label,
                "session_name": LABEL_TO_SESSION_NAME.get(label, label),
                "base_match": {
                    "match_id": m.match_id,
                    "match_date": m.match_date.isoformat(),
                    "matchday": m.matchday,
                    "opponent": m.opponent,
                    "home_away": m.home_away,
                },
                "load_factor": factor,
                "base_match_total_distance_gross_km": base,
                "planned_total_distance_gross_km": planned,
                "contents": contents,
                "notes": None,
            }
        )

    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Create training_schedule.json from DFL matchdays.")
    parser.add_argument(
        "--team-matches-json",
        default="dfl_running_team_matches.json",
        help="Path to dfl_running_team_matches.json",
    )
    parser.add_argument("--team", default="Hamburger SV", help="Team name as in the DFL JSON")
    parser.add_argument("--out", default="training_schedule.json", help="Output JSON path")
    args = parser.parse_args()

    team_matches_path = Path(args.team_matches_json)
    matches = load_matches_for_team(team_matches_path, args.team)

    schedule = build_schedule(matches)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(schedule, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(schedule)} sessions to {out_path}")


if __name__ == "__main__":
    main()

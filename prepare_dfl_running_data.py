import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _canon(text: Any) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()
    # Pandas may decode some umlauts badly into the replacement char.
    s = s.replace("\ufffd", "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if ch.isalnum())
    return s


def _parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "-"}:
        return None

    # Percent strings like "47,37%" (we generally don't aggregate these)
    s = s.replace("%", "")

    # Keep digits, comma, dot, minus
    s = re.sub(r"[^0-9,\.\-]", "", s)
    if not s:
        return None

    # German decimals
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    # If both are present, assume "." thousands and "," decimals
    elif s.count(",") >= 1 and s.count(".") >= 1:
        s = s.replace(".", "").replace(",", ".")

    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):
        try:
            # datetime/date
            return value.isoformat()[:10]
        except Exception:
            pass
    s = str(value).strip()
    if not s:
        return None
    # Common formats: 2025-08-24, 24.08.2025
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        return s
    m = re.match(r"^(\d{2})\.(\d{2})\.(\d{4})$", s)
    if m:
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}-{mm}-{dd}"
    return s


@dataclass(frozen=True)
class MetricSpec:
    key: str
    candidates: Tuple[str, ...]
    agg: str  # 'sum' | 'max'


MATCH_KEY_CANDIDATES = {
    "match_id": ("Spiel ID", "SpielID", "Match ID", "MatchID"),
    "date": ("Datum", "Date"),
    "matchday": ("Spieltag", "Matchday"),
    "team": ("Team",),
    "opponent": ("Gegner", "Opponent"),
    "home_away": ("H/A", "Home/Away", "Heim/Auswaerts"),
    "goals_for": ("T (Team)", "Tore (Team)", "Tore Team"),
    "goals_against": ("GT", "Gegentore", "Gegentore (Team)"),
}


METRICS: List[MetricSpec] = [
    # Gross (match clock) totals
    MetricSpec("total_distance_gross_km", ("Gesamtlaufdistanz (km)", "Gesamtlaufdistanz"), "sum"),
    MetricSpec("intense_distance_gross_km", ("Distanz Intensive Läufe (km)", "Distanz Intensive Laeufe (km)", "Distanz Intensive Läufe"), "sum"),

    MetricSpec("total_distance_net_km", ("Gesamtlaufdistanz Nettospielzeit (km)",), "sum"),
    MetricSpec("max_speed_kmh", ("Maximaltempo (km/h)",), "max"),
    MetricSpec("sprint_distance_net_km", ("Sprintdistanz Nettospielzeit (km)",), "sum"),
    MetricSpec("sprints_net", ("Sprints Nettospielzeit",), "sum"),
    MetricSpec("sprints_in_sr", ("Sprints in SR",), "sum"),
    MetricSpec("sprints_vs_sr", ("Sprints gg. SR", "Sprints gg SR"), "sum"),
    MetricSpec("tempo_runs_net", ("Tempoläufe Nettospielzeit",), "sum"),
    MetricSpec("tempo_distance_net_km", ("Distanz Tempoläufe Nettospielzeit (km)",), "sum"),
    MetricSpec("intense_runs_net", ("Intensive Läufe Nettospielzeit", "Intensive Läufe nettospielzeit"), "sum"),
    MetricSpec("intense_distance_net_km", ("Distanz Intensive Läufe Nettospielzeit (km)", "DEistanz intensive Läufe Nettospielzeit"), "sum"),
    MetricSpec("intense_runs_in_sr", ("Intensive Läufe in SR",), "sum"),
    MetricSpec("intense_runs_vs_sr", ("Intensive Läufe gg. SR", "Intensive Läufe gg SR"), "sum"),
    MetricSpec("tempo_runs_in_sr", ("Tempoläufe in SR",), "sum"),
    MetricSpec("tempo_runs_vs_sr", ("Tempoläufe gg. SR", "Tempoläufe gg SR"), "sum"),

    # In Ballbesitz (BB)
    MetricSpec("total_distance_possession_km", ("Gesamtlaufdistanz in BB (km)",), "sum"),
    MetricSpec("sprints_possession", ("Sprints in BB",), "sum"),
    MetricSpec("sprint_distance_possession_km", ("Sprintdistanz in BB (km)",), "sum"),
    MetricSpec("tempo_runs_possession", ("Tempoläufe in BB",), "sum"),
    MetricSpec("tempo_distance_possession_km", ("Distanz Tempoläufe in BB (km)",), "sum"),
    MetricSpec("intense_runs_possession", ("Intensive Läufe in BB",), "sum"),
    MetricSpec("intense_distance_possession_km", ("Distanz Intensive Läufe in BB (km)",), "sum"),

    # Gegen den Ball
    MetricSpec("total_distance_outpos_km", ("Gesamtlaufdistanz gg. den Ball (km)", "Gesamtlaufdistanz gg. den Ball"), "sum"),
    MetricSpec("sprints_outpos", ("Sprints gg. den Ball",), "sum"),
    MetricSpec("sprint_distance_outpos_km", ("Sprintdistanz gg. den Ball (km)",), "sum"),
    MetricSpec("tempo_runs_outpos", ("Tempoläufe gg. den Ball",), "sum"),
    MetricSpec("tempo_distance_outpos_km", ("Distanz Tempoläufe gg. den Ball (km)",), "sum"),
    MetricSpec("intense_runs_outpos", ("Intensive Läufe gg. den Ball",), "sum"),
    MetricSpec("intense_distance_outpos_km", ("Distanz Intensive Läufe gg. den Ball (km)",), "sum"),
]


def _resolve_columns(df: pd.DataFrame, wanted: Iterable[Tuple[str, Tuple[str, ...]]]) -> Dict[str, str]:
    col_map: Dict[str, str] = {}
    canon_to_actual: Dict[str, str] = {}
    for c in df.columns:
        canon_to_actual[_canon(c)] = c

    for out_key, candidates in wanted:
        chosen: Optional[str] = None
        for cand in candidates:
            cc = _canon(cand)
            if cc in canon_to_actual:
                chosen = canon_to_actual[cc]
                break
        if chosen:
            col_map[out_key] = chosen
    return col_map


def _load_one_xlsx(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    # DFL exports usually use this sheet name; fallback to first.
    sheet = "Spieler Statistiken" if "Spieler Statistiken" in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    df["__source_file"] = path.name
    df["__source_sheet"] = sheet
    return df


def _prepare_player_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleanup: drop fully empty rows
    df = df.dropna(how="all").copy()

    # Normalize key columns
    keys_map = _resolve_columns(df, MATCH_KEY_CANDIDATES.items())
    for must in ["match_id", "date", "team", "opponent"]:
        if must not in keys_map:
            raise ValueError(f"Required column not found in DFL file: {must} (candidates={MATCH_KEY_CANDIDATES[must]})")

    # Optional minutes columns
    # - minutes: gross (match clock) minutes played
    # - minutes_net: net playing time minutes (ball-in-play) if available in the export
    minutes_col = None
    for cand in ("Spielminuten", "Minuten", "Minutes"):
        cc = _canon(cand)
        for c in df.columns:
            if _canon(c) == cc:
                minutes_col = c
                break
        if minutes_col:
            break

    minutes_net_col = None
    for cand in (
        "Nettominuten",
        "Netto Minuten",
        "Netto-Minuten",
        "Nettospielzeit Minuten",
        "Nettospielzeit (min)",
        "Net playing time (min)",
        "Net playing time minutes",
    ):
        cc = _canon(cand)
        for c in df.columns:
            if _canon(c) == cc:
                minutes_net_col = c
                break
        if minutes_net_col:
            break

    # Extract / parse match keys
    out = pd.DataFrame()
    out["match_id"] = df[keys_map["match_id"]].astype(str)
    out["match_date"] = df[keys_map["date"]].map(_parse_date)
    out["matchday"] = df[keys_map.get("matchday")] if "matchday" in keys_map else None
    out["team"] = df[keys_map["team"]].astype(str)
    out["opponent"] = df[keys_map["opponent"]].astype(str)
    out["home_away"] = df[keys_map.get("home_away")] if "home_away" in keys_map else None
    out["goals_for"] = df[keys_map.get("goals_for")] if "goals_for" in keys_map else None
    out["goals_against"] = df[keys_map.get("goals_against")] if "goals_against" in keys_map else None

    # Player identity
    first = None
    last = None
    for cand in ("Vorname", "First name", "FirstName"):
        for c in df.columns:
            if _canon(c) == _canon(cand):
                first = c
                break
        if first:
            break
    for cand in ("Nachname", "Last name", "LastName"):
        for c in df.columns:
            if _canon(c) == _canon(cand):
                last = c
                break
        if last:
            break

    if first and last:
        out["player_name"] = (df[first].astype(str).str.strip() + " " + df[last].astype(str).str.strip()).str.strip()
    else:
        # fallback
        pn = None
        for cand in ("Spieler", "Player", "Name"):
            for c in df.columns:
                if _canon(c) == _canon(cand):
                    pn = c
                    break
            if pn:
                break
        out["player_name"] = df[pn].astype(str) if pn else None

    if minutes_col:
        out["minutes"] = df[minutes_col].map(_parse_number)
    else:
        out["minutes"] = None

    if minutes_net_col:
        out["minutes_net"] = df[minutes_net_col].map(_parse_number)
    else:
        out["minutes_net"] = None

    out["source_file"] = df["__source_file"].astype(str)
    out["source_sheet"] = df["__source_sheet"].astype(str)

    # Metrics
    metric_cols_wanted = [(m.key, m.candidates) for m in METRICS]
    metric_map = _resolve_columns(df, metric_cols_wanted)
    for m in METRICS:
        if m.key not in metric_map:
            out[m.key] = None
            continue
        out[m.key] = df[metric_map[m.key]].map(_parse_number)

    # Drop players with 0 minutes (optional)
    if minutes_col:
        out = out[(out["minutes"].fillna(0) > 0)].copy()
    elif minutes_net_col:
        out = out[(out["minutes_net"].fillna(0) > 0)].copy()

    return out


def _aggregate_team(rows: pd.DataFrame, team_name: Optional[str]) -> pd.DataFrame:
    # If team_name is None/ALL, aggregate all teams contained in the export.
    if team_name and str(team_name).strip().upper() not in {"ALL", "*"}:
        df = rows[rows["team"] == team_name].copy()
    else:
        df = rows.copy()

    # Aggregations
    sum_keys = [m.key for m in METRICS if m.agg == "sum"]
    max_keys = [m.key for m in METRICS if m.agg == "max"]

    group_keys = ["match_id", "match_date", "matchday", "team", "opponent", "home_away", "goals_for", "goals_against"]

    agg: Dict[str, Any] = {k: "sum" for k in sum_keys}
    for k in max_keys:
        agg[k] = "max"
    # informative
    # Use sum(min_count=1) so a fully-missing column stays NaN (instead of 0)
    agg["minutes"] = lambda s: s.sum(min_count=1)
    if "minutes_net" in df.columns:
        agg["minutes_net"] = lambda s: s.sum(min_count=1)

    g = df.groupby(group_keys, dropna=False).agg(agg).reset_index()

    # Sort by date if possible
    if "match_date" in g.columns:
        g = g.sort_values(["match_date", "match_id"], na_position="last")

    # add match_label
    def _label(r: pd.Series) -> str:
        d = r.get("match_date") or ""
        opp = r.get("opponent") or ""
        ha = r.get("home_away") or ""
        return f"{d} | {ha} vs {opp}".strip()

    g["match_label"] = g.apply(_label, axis=1)

    # Derived match-minute fields (per team)
    # In football, sum of player minutes ~= 11 * match minutes.
    def _div11(v: Any) -> Optional[float]:
        x = _parse_number(v)
        if x is None or x <= 0:
            return None
        return x / 11.0

    g["match_minutes_gross"] = g["minutes"].map(_div11) if "minutes" in g.columns else None
    # If a real net-minutes column is present in the export, use it.
    # Otherwise, fall back to match minutes derived from player minutes (minutes/11).
    if "minutes_net" in g.columns:
        net_from_export = g["minutes_net"].map(_div11)
        gross = g["match_minutes_gross"]
        g["match_minutes_net"] = net_from_export.where(net_from_export.notna(), gross)
    else:
        g["match_minutes_net"] = g["match_minutes_gross"]

    return g


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare DFL running metrics (xlsx) for analysis.")
    ap.add_argument("--input-dir", default="Data/dfl", help="Folder containing DFL xlsx exports")
    ap.add_argument("--team", default="ALL", help="Team name to aggregate (use ALL to include both teams per match)")
    ap.add_argument("--out-dir", default=".", help="Output folder")
    ap.add_argument("--write-player-json", action="store_true", help="Also write per-player match rows")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.xlsx"))
    if not files:
        raise SystemExit(f"No .xlsx files found in {in_dir}")

    all_frames: List[pd.DataFrame] = []
    for fp in files:
        df = _load_one_xlsx(fp)
        all_frames.append(df)

    merged = pd.concat(all_frames, ignore_index=True)
    player_rows = _prepare_player_rows(merged)

    # When using many team exports (e.g., --team ALL), the same match can appear in multiple
    # xlsx files and therefore the per-player rows can be duplicated. If we don't de-duplicate
    # here, team aggregations will be summed multiple times.
    before = len(player_rows)
    player_rows = player_rows.drop_duplicates(subset=["match_id", "team", "player_name"], keep="first").copy()
    dropped = before - len(player_rows)
    if dropped:
        print(f"Deduplicated player rows: dropped {dropped} duplicates")

    team_rows = _aggregate_team(player_rows, args.team)

    # Output files
    team_json_path = out_dir / "dfl_running_team_matches.json"
    team_csv_path = out_dir / "dfl_running_team_matches.csv"
    team_rows.to_csv(team_csv_path, index=False, encoding="utf-8")
    team_json_path.write_text(team_rows.to_json(orient="records", force_ascii=False), encoding="utf-8")

    if args.write_player_json:
        player_json_path = out_dir / "dfl_running_players.json"
        player_csv_path = out_dir / "dfl_running_players.csv"
        player_rows.to_csv(player_csv_path, index=False, encoding="utf-8")
        player_json_path.write_text(player_rows.to_json(orient="records", force_ascii=False), encoding="utf-8")

    print(f"Wrote: {team_json_path} and {team_csv_path}")
    if args.write_player_json:
        print("Wrote per-player outputs too")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

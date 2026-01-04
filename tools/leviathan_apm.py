import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SHOT_TYPES = {"Shot", "MissedShots", "SavedShot", "Goal", "ShotOnPost"}


def iter_json_array_objects(path: Path):
    """Stream objects from a top-level JSON array without loading the full file."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        buf = ""
        # Seek '['
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                raise RuntimeError("Unexpected EOF while searching for '['")
            buf += chunk
            i = buf.find("[")
            if i != -1:
                buf = buf[i + 1 :]
                break

        while True:
            # Skip whitespace/commas
            j = 0
            while j < len(buf) and buf[j] in " \t\r\n,":
                j += 1
            buf = buf[j:]

            if buf.startswith("]"):
                return

            # Decode next object
            while True:
                try:
                    obj, idx = decoder.raw_decode(buf)
                    buf = buf[idx:]
                    yield obj
                    break
                except json.JSONDecodeError:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        raise
                    buf += chunk


def shot_xg_from_events_row(e: dict) -> float | None:
    """Approximate the same xG model used in kennzahlen.py, using only columns available in events_data.json."""
    try:
        x_coord = float(e.get("x_coord"))
        y_coord = float(e.get("y_coord"))
    except (TypeError, ValueError):
        return None

    # Field scaling to meters
    x_m = x_coord / 100.0 * 105.0
    y_m = y_coord / 100.0 * 68.0

    goal_w = 7.32
    x_dist = max(1e-6, 105.0 - x_m)
    y_off = abs(y_m - 34.0)
    dist_m = math.sqrt(x_dist * x_dist + y_off * y_off)

    denom = (x_dist * x_dist + y_off * y_off - (goal_w / 2.0) ** 2)
    angle_rad = math.pi if denom <= 0 else math.atan((goal_w * x_dist) / denom)

    in_box = (x_m >= (105.0 - 16.5)) and (y_off <= 20.16)
    close_range = dist_m <= 12.0

    # Use boolean flags (events_data.json does not contain qualifiers)
    is_pen = bool(e.get("is_penalty") is True)
    is_header = bool(e.get("is_header") is True)
    is_from_corner = bool(e.get("is_corner") is True)
    is_fastbreak = False  # not available
    is_blocked = False  # not available
    is_cross = bool(e.get("is_cross") is True)

    # Logit base
    b0, b1, b2 = -2.15, -0.060, 1.55
    logit = b0 + b1 * dist_m + b2 * angle_rad

    # Adjustments (same as kennzahlen.py where possible)
    logit += 0.0 if in_box else -0.35
    logit += 0.25 if close_range else 0.0
    logit += -0.45 if is_header else 0.0
    logit += -0.25 if is_from_corner else 0.0
    logit += 0.12 if is_fastbreak else 0.0
    # Out-of-box / box-loc not available
    logit += -0.10 if is_cross else 0.0
    if dist_m >= 35.0:
        logit += -0.70
    elif dist_m >= 30.0:
        logit += -0.50

    xg = 1.0 / (1.0 + math.exp(-logit))
    if is_blocked:
        xg *= 0.92
    if is_pen:
        xg = 0.76
    xg = min(0.97, max(0.001, xg))
    return float(xg)


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    lineup: tuple[int, ...]  # sorted player IDs on pitch
    team: str


def lineup_from_formation_row(r: dict) -> tuple[int, ...]:
    ids = r.get("playerIds") or []
    slots = r.get("formationSlots") or []
    out: list[int] = []

    # Prefer slots 1..11, else fallback to first 11 ids.
    if isinstance(ids, list) and isinstance(slots, list) and len(ids) == len(slots) and len(ids) > 0:
        for pid, slot in zip(ids, slots):
            try:
                s = int(slot)
            except Exception:
                continue
            if 1 <= s <= 11:
                try:
                    out.append(int(pid))
                except Exception:
                    continue
    elif isinstance(ids, list):
        for pid in ids[:11]:
            try:
                out.append(int(pid))
            except Exception:
                continue

    # Defensive: ensure we only keep 11-ish players
    out = out[:11]
    out = sorted(set(out))
    return tuple(out)


def find_lineup_at(intervals: list[Interval], t: float) -> tuple[int, ...] | None:
    for it in intervals:
        if it.start <= t < it.end:
            return it.lineup
    # If t is exactly at the end, try the last interval
    for it in reversed(intervals):
        if abs(t - it.end) < 1e-9:
            return it.lineup
    return None


def sum_shots_between(shots: list[tuple[float, float]], start: float, end: float) -> float:
    if not shots:
        return 0.0
    s = 0.0
    for t, xg in shots:
        if start <= t < end:
            s += xg
    return s


def load_processed_team_xg(processed_path: Path) -> dict[tuple[str, str], float]:
    data = json.loads(processed_path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], float] = defaultdict(float)
    for r in data if isinstance(data, list) else []:
        if not isinstance(r, dict):
            continue
        mid = str(r.get("match_id") or "")
        team = str(r.get("team") or "")
        if not mid or not team:
            continue
        # Prefer full-match rows (half==0). If half is missing, accept.
        if ("half" in r) and (r.get("half") is not None):
            try:
                if int(r.get("half")) != 0:
                    continue
            except Exception:
                pass
        try:
            xg = float(r.get("xg") or 0.0)
        except Exception:
            xg = 0.0
        if math.isfinite(xg):
            out[(mid, team)] += xg
    return dict(out)


def load_processed_players_for_team(processed_path: Path, team: str) -> set[str]:
    data = json.loads(processed_path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for r in data if isinstance(data, list) else []:
        if not isinstance(r, dict):
            continue
        if str(r.get("team") or "") != team:
            continue
        # Prefer full-match rows (half==0). If half is missing, accept.
        if ("half" in r) and (r.get("half") is not None):
            try:
                if int(r.get("half")) != 0:
                    continue
            except Exception:
                pass
        name = str(r.get("player") or "").strip()
        if name:
            out.add(name)
    return out


def main():
    ap = argparse.ArgumentParser(description="Leviathan-style ridge adjusted plus-minus from formations + events + processed xG.")
    ap.add_argument("--lambda", dest="lam", type=float, default=50.0, help="Ridge regularization strength (default: 50)")
    ap.add_argument("--min-seg-minutes", type=float, default=0.5, help="Minimum segment length in minutes (default: 0.5)")
    ap.add_argument("--max-matches", type=int, default=0, help="If >0, only process the first N matches (for quick tests)")
    ap.add_argument("--ui-team", type=str, default="Hamburg", help="Team name as used in processed_data.json for the UI mapping (default: Hamburg)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    formations_path = root / "formations_data.json"
    events_path = root / "events_data.json"
    processed_path = root / "processed_data.json"

    if not formations_path.exists():
        raise SystemExit(f"Missing: {formations_path}")
    if not events_path.exists():
        raise SystemExit(f"Missing: {events_path}")
    if not processed_path.exists():
        raise SystemExit(f"Missing: {processed_path}")

    print("Loading formations…")
    formations = json.loads(formations_path.read_text(encoding="utf-8"))
    if not isinstance(formations, list):
        raise SystemExit("formations_data.json must be a JSON array")

    print("Indexing processed team xG (same source as Spielanalyse Übersicht)…")
    processed_team_xg = load_processed_team_xg(processed_path)
    ui_team = str(args.ui_team)
    ui_team_players = load_processed_players_for_team(processed_path, ui_team)

    print("Streaming events and collecting shots…")
    shots_raw: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    player_id_to_name: dict[int, str] = {}

    for e in iter_json_array_objects(events_path):
        if not isinstance(e, dict):
            continue

        # playerId -> name map
        pid = e.get("playerId")
        pname = e.get("player_name")
        if pid is not None and pname:
            try:
                ipid = int(float(pid))
                if ipid not in player_id_to_name:
                    player_id_to_name[ipid] = str(pname)
            except Exception:
                pass

        et = str(e.get("event_type") or "")
        is_shot = (e.get("isShot") is True) or (et in SHOT_TYPES)
        if not is_shot:
            continue

        mid = str(e.get("match_id") or "")
        team = str(e.get("team") or "")
        if not mid or not team:
            continue

        try:
            minute = float(e.get("minute") or 0.0)
            second = float(e.get("second") or 0.0)
        except Exception:
            minute, second = 0.0, 0.0
        t_min = float(minute + (second / 60.0))

        xg = shot_xg_from_events_row(e)
        if xg is None or not math.isfinite(xg):
            continue

        shots_raw[(mid, team)].append((t_min, xg))

    # Sort shots per match/team
    for k in list(shots_raw.keys()):
        shots_raw[k].sort(key=lambda p: p[0])

    # Scale raw shot-xG to match processed_data.json totals (so we use the same xG shown in Spielanalyse)
    shots_scaled: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for (mid, team), shots in shots_raw.items():
        raw_sum = sum(x for _, x in shots)
        target = processed_team_xg.get((mid, team))
        if target is None or not math.isfinite(target):
            scale = 1.0
        else:
            scale = (target / raw_sum) if (raw_sum > 1e-9) else 0.0
        shots_scaled[(mid, team)] = [(t, x * scale) for (t, x) in shots]

    print("Building formation intervals…")
    by_match: dict[str, dict[str, list[Interval]]] = defaultdict(lambda: {"home": [], "away": []})

    for r in formations:
        if not isinstance(r, dict):
            continue
        mid = str(r.get("match_id") or "")
        field = str(r.get("field") or "")
        team = str(r.get("team_name") or "")
        if not mid or field not in {"home", "away"} or not team:
            continue
        try:
            start = float(r.get("startMinuteExpanded") or 0.0)
            end = float(r.get("endMinuteExpanded") or 0.0)
        except Exception:
            continue
        if not (end > start):
            continue
        lineup = lineup_from_formation_row(r)
        if len(lineup) < 7:  # sanity (red cards still should be >=10 usually, but be lenient)
            continue
        by_match[mid][field].append(Interval(start=start, end=end, lineup=lineup, team=team))

    # Sort intervals
    match_ids = sorted(by_match.keys())
    if args.max_matches and args.max_matches > 0:
        match_ids = match_ids[: args.max_matches]

    rows_home: list[tuple[tuple[int, ...], tuple[int, ...], float, float]] = []
    # (home_lineup, away_lineup, y_rate_per90, minutes)

    # Track minutes per player on-pitch
    minutes_by_player: dict[int, float] = defaultdict(float)

    print("Creating constant-lineup segments and aggregating xG…")
    for mid in match_ids:
        home_int = sorted(by_match[mid]["home"], key=lambda it: it.start)
        away_int = sorted(by_match[mid]["away"], key=lambda it: it.start)
        if not home_int or not away_int:
            continue

        home_team = home_int[0].team
        away_team = away_int[0].team

        bounds = sorted({it.start for it in home_int} | {it.end for it in home_int} | {it.start for it in away_int} | {it.end for it in away_int})
        if len(bounds) < 2:
            continue

        shots_h = shots_scaled.get((mid, home_team), [])
        shots_a = shots_scaled.get((mid, away_team), [])

        for a, b in zip(bounds, bounds[1:]):
            dur = b - a
            if dur < args.min_seg_minutes:
                continue
            mid_t = (a + b) / 2.0
            hl = find_lineup_at(home_int, mid_t)
            al = find_lineup_at(away_int, mid_t)
            if not hl or not al:
                continue

            xg_home = sum_shots_between(shots_h, a, b)
            xg_away = sum_shots_between(shots_a, a, b)
            y = xg_home - xg_away
            y_rate = (y / dur) * 90.0 if dur > 1e-9 else 0.0

            rows_home.append((hl, al, float(y_rate), float(dur)))

            for pid in hl:
                minutes_by_player[pid] += dur
            for pid in al:
                minutes_by_player[pid] += dur

    if not rows_home:
        raise SystemExit("No segments built. Check formations_data.json coverage.")

    # Build player index
    player_ids = sorted({pid for hl, al, _, _ in rows_home for pid in hl} | {pid for hl, al, _, _ in rows_home for pid in al})
    p_index = {pid: i for i, pid in enumerate(player_ids)}

    n = len(rows_home)
    p = len(player_ids)
    print(f"Segments: {n} | Players: {p}")

    # Build X, y, w
    X = np.zeros((n, p + 1), dtype=np.float32)  # + intercept
    y = np.zeros((n,), dtype=np.float32)
    w = np.zeros((n,), dtype=np.float32)

    for i, (hl, al, y_rate, dur) in enumerate(rows_home):
        for pid in hl:
            X[i, p_index[pid]] = 1.0
        for pid in al:
            X[i, p_index[pid]] = -1.0
        X[i, p] = 1.0
        y[i] = y_rate
        w[i] = dur

    sw = np.sqrt(np.maximum(w, 1e-9)).astype(np.float32)
    Xw = X * sw[:, None]
    yw = y * sw

    lam = float(args.lam)
    # Do not penalize intercept
    ridge = np.full((p + 1,), lam, dtype=np.float32)
    ridge[p] = 0.0

    A = (Xw.T @ Xw).astype(np.float64)
    A[np.diag_indices_from(A)] += ridge.astype(np.float64)
    b = (Xw.T @ yw).astype(np.float64)

    print("Solving ridge…")
    beta = np.linalg.solve(A, b).astype(np.float64)

    out_rows = []
    for pid in player_ids:
        name = player_id_to_name.get(pid, "")
        out_rows.append(
            {
                "playerId": pid,
                "player_name": name,
                "xG_diff_per90": float(beta[p_index[pid]]),
                "minutes": float(minutes_by_player.get(pid, 0.0)),
            }
        )

    out_rows.sort(key=lambda r: r["xG_diff_per90"], reverse=True)

    out_path = root / "leviathan_apm_results.csv"
    print(f"Writing {out_path}…")
    import csv

    with out_path.open("w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["playerId", "player_name", "xG_diff_per90", "minutes"])
        wtr.writeheader()
        wtr.writerows(out_rows)

    out_json_path = root / "leviathan_apm_results.json"
    print(f"Writing {out_json_path}…")
    out_json_path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # UI mapping for Übersicht: player name -> impact per90
    ui_map: dict[str, float] = {}
    for r in out_rows:
        nm = str(r.get("player_name") or "").strip()
        if not nm:
            continue
        if ui_team_players and (nm not in ui_team_players):
            continue
        try:
            v = float(r.get("xG_diff_per90"))
        except Exception:
            continue
        if math.isfinite(v):
            ui_map[nm] = v

    ui_map_path = root / "leviathan_apm_hsv_map.json"
    print(f"Writing {ui_map_path}…")
    ui_map_path.write_text(json.dumps(ui_map, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")


if __name__ == "__main__":
    main()

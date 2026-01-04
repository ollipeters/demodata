import csv
import json
import math
import random
import uuid
import argparse
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, time, timedelta, timezone
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parent
SCHEDULE_PATH = WORKSPACE_DIR / "training_schedule.json"
PLAYERS_CSV_PATH = WORKSPACE_DIR / "dfl_running_players.csv"
TEMPLATE_GPS_PATH = WORKSPACE_DIR / "Data" / "GPS" / "20240316_27c44cee-5332-4111-ad14-42965dee59c8.csv"
OUTPUT_CSV_PATH = WORKSPACE_DIR / "fake_gps_training_cuts.csv"

TEAM_NAME = "Hamburger SV"

# Standardisierte Aktivierungen: bewusst begrenzter Pool, der sich wiederholen kann.
# Wichtig: Namen enthalten "Aktiv" damit die UI sie sicher als Aktivierung kategorisieren kann.
ACTIVATION_LIBRARY: list[str] = [
    "Aktivierung 01",
    "Aktivierung 02",
    "Aktivierung 03",
    "Aktivierung 04",
    "Aktivierung 05",
    "Aktivierung 06",
    "Aktivierung 07",
    "Aktivierung 08",
    "Aktivierung 09",
    "Aktivierung 10",
    "Aktivierung 11",
    "Aktivierung 12",
    "Aktivierung 13",
    "Aktivierung 14",
    "Aktivierung 15",
    "Aktivierung 16",
    "Aktivierung 17",
    "Aktivierung 18",
    "Aktivierung 19",
    "Aktivierung 20",
    "Aktivierung 21",
    "Aktivierung 22",
]

# Unregelmäßige Wiederholung: wenige Aktivierungen kommen häufiger vor.
ACTIVATION_WEIGHTS: list[float] = [
    8.0,
    7.0,
    6.0,
    6.0,
    5.5,
    5.0,
    4.5,
    4.0,
    4.0,
    3.8,
    3.6,
    3.4,
    3.2,
    3.0,
    2.8,
    2.6,
    2.4,
    2.2,
    2.0,
    1.8,
    1.6,
    1.4,
]


def _weighted_choice(items: list[str], weights: list[float], rng: random.Random) -> str:
    if not items:
        return ""
    if len(items) != len(weights):
        return rng.choice(items)
    total = sum(max(0.0, float(w)) for w in weights)
    if total <= 0:
        return rng.choice(items)
    r = rng.random() * total
    acc = 0.0
    for it, w in zip(items, weights):
        acc += max(0.0, float(w))
        if r <= acc:
            return it
    return items[-1]


def standardized_activation_name(d: Date, label: str, original_name: str) -> str:
    """Return a stable (per day+label+original) activation name from a fixed pool.

    This keeps the exercise library realistic: a small set of standardized activations
    that repeat irregularly across sessions.
    """
    seed = uuid.uuid5(uuid.NAMESPACE_DNS, f"hsv-activation::{d.isoformat()}::{label}::{original_name}").int & 0xFFFFFFFF
    rr = random.Random(seed)
    return _weighted_choice(ACTIVATION_LIBRARY, ACTIVATION_WEIGHTS, rr)


def _safe_text(value: str) -> str:
    if value is None:
        return ""
    return str(value).replace("\uFFFD", "")


def _parse_iso_date(d: str) -> Date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def _format_statsports_date(d: Date) -> str:
    # template uses dd/mm/yyyy
    return d.strftime("%d/%m/%Y")


def _epoch_seconds_local_noon(d: Date) -> float:
    # deterministic but simple; StatsSports template uses unix seconds
    dt = datetime.combine(d, time(12, 0, 0)).replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _uuid_for_player(player_name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"hsv-player::{player_name}"))


def _uuid_for_activity(d: Date, label: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"hsv-activity::{d.isoformat()}::{label}"))


def _uuid_for_period(d: Date, label: str, period_name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"hsv-period::{d.isoformat()}::{label}::{period_name}"))


def _dirichlet(alpha: list[float], rng: random.Random) -> list[float]:
    # no numpy dependency
    draws = [rng.gammavariate(a, 1.0) for a in alpha]
    s = sum(draws)
    if s <= 0:
        return [1.0 / len(alpha)] * len(alpha)
    return [x / s for x in draws]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class Baseline:
    # medians from the template (used as anchors)
    total_duration_s: float
    total_distance_m: float
    meter_per_min: float
    total_player_load: float
    player_load_per_min: float
    peak_player_load: float
    max_vel: float
    vel_share_1to5: list[float]
    decel_per_min: float
    max_effort_decel: float


def read_template_baseline() -> Baseline:
    import pandas as pd

    use_cols = [
        "total_duration",
        "total_distance",
        "meterage_per_minute",
        "total_player_load",
        "player_load_per_minute",
        "peak_player_load",
        "max_vel",
        "deceleration_per_minute",
        "max_effort_deceleration",
    ] + [f"velocity_band{i}_total_distance" for i in range(1, 6)]

    header = pd.read_csv(TEMPLATE_GPS_PATH, nrows=0).columns
    use = [c for c in use_cols if c in header]
    df = pd.read_csv(TEMPLATE_GPS_PATH, usecols=use)

    def med(col: str, default: float) -> float:
        if col not in df.columns:
            return default
        s = df[col].dropna()
        if len(s) == 0:
            return default
        return float(s.median())

    total_duration_s = med("total_duration", 3000.0)
    total_distance_m = med("total_distance", 5000.0)
    meter_per_min = med("meterage_per_minute", 95.0)
    total_player_load = med("total_player_load", 500.0)
    player_load_per_min = med("player_load_per_minute", 10.0)
    peak_player_load = med("peak_player_load", 7.0)
    max_vel = med("max_vel", 28.0)
    decel_per_min = med("deceleration_per_minute", 1.0)
    max_effort_decel = med("max_effort_deceleration", -5.0)

    vel_cols = [f"velocity_band{i}_total_distance" for i in range(1, 6) if f"velocity_band{i}_total_distance" in df.columns]
    if vel_cols and "total_distance" in df.columns:
        td = df["total_distance"].replace(0, math.nan)
        shares_df = df[vel_cols].div(td, axis=0)
        vel_share_1to5 = [float(shares_df[c].median()) for c in vel_cols]
        # if anything missing, pad
        while len(vel_share_1to5) < 5:
            vel_share_1to5.append(0.0)
        # normalize
        s = sum(vel_share_1to5)
        vel_share_1to5 = [x / s for x in vel_share_1to5] if s > 0 else [0.34, 0.45, 0.10, 0.06, 0.05]
    else:
        vel_share_1to5 = [0.34, 0.45, 0.10, 0.06, 0.05]

    return Baseline(
        total_duration_s=total_duration_s,
        total_distance_m=total_distance_m,
        meter_per_min=meter_per_min,
        total_player_load=total_player_load,
        player_load_per_min=player_load_per_min,
        peak_player_load=peak_player_load,
        max_vel=max_vel,
        vel_share_1to5=vel_share_1to5,
        decel_per_min=decel_per_min,
        max_effort_decel=max_effort_decel,
    )


def default_baseline() -> Baseline:
    # Anchors taken from the example file medians we previously observed.
    return Baseline(
        total_duration_s=3048.63,
        total_distance_m=5046.34,
        meter_per_min=97.74,
        total_player_load=521.54,
        player_load_per_min=9.89,
        peak_player_load=7.09,
        max_vel=28.64,
        vel_share_1to5=[0.34, 0.45, 0.10, 0.06, 0.05],
        decel_per_min=1.015,
        max_effort_decel=-5.079,
    )


def load_players(team_name: str) -> list[str]:
    import pandas as pd

    df = pd.read_csv(PLAYERS_CSV_PATH)
    players = sorted(df.loc[df["team"] == team_name, "player_name"].dropna().unique().tolist())
    if not players:
        raise RuntimeError(f"No players found for team={team_name} in {PLAYERS_CSV_PATH}")
    return players


def load_schedule() -> list[dict]:
    return json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))


def content_profile(content_type: str) -> dict:
    # Defines how intensity shifts zone shares and accel/decel density
    t = (content_type or "").lower()
    if t == "match":
        # Full match load: should dominate the day vs. activation
        return {
            "dist_w": 12.0,
            "dur_w": 14.0,
            "vel_mult": [0.5, 0.9, 1.2, 1.3, 1.1],
            "acc_mult": 1.2,
            "dec_mult": 1.2,
            "maxv_mult": 1.05,
        }
    if t == "activation":
        return {"dist_w": 0.9, "dur_w": 1.0, "vel_mult": [1.8, 1.2, 0.7, 0.4, 0.2], "acc_mult": 0.5, "dec_mult": 0.6, "maxv_mult": 0.85}
    if t == "rondo":
        return {"dist_w": 1.0, "dur_w": 1.0, "vel_mult": [1.2, 1.3, 1.0, 0.6, 0.3], "acc_mult": 0.9, "dec_mult": 1.0, "maxv_mult": 0.9}
    if t in ("ssg", "game"):
        return {"dist_w": 1.15, "dur_w": 1.0, "vel_mult": [0.7, 0.9, 1.2, 1.2, 1.1], "acc_mult": 1.4, "dec_mult": 1.5, "maxv_mult": 1.0}
    if t == "tactical":
        return {"dist_w": 1.0, "dur_w": 1.05, "vel_mult": [1.0, 1.1, 1.0, 0.8, 0.5], "acc_mult": 0.8, "dec_mult": 0.9, "maxv_mult": 0.92}
    if t == "athletics":
        return {"dist_w": 0.95, "dur_w": 0.9, "vel_mult": [0.6, 0.8, 1.0, 1.3, 1.6], "acc_mult": 1.2, "dec_mult": 1.0, "maxv_mult": 1.1}
    return {"dist_w": 1.0, "dur_w": 1.0, "vel_mult": [1, 1, 1, 1, 1], "acc_mult": 1.0, "dec_mult": 1.0, "maxv_mult": 1.0}


def allocate(weights: list[float]) -> list[float]:
    s = sum(weights)
    if s <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / s for w in weights]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate fake StatsSports-like GPS KPIs per training day/content/player")
    p.add_argument("--team", default=TEAM_NAME, help="Team name to use for player list")
    p.add_argument("--out", default=str(OUTPUT_CSV_PATH), help="Output CSV path")
    p.add_argument("--seed", type=int, default=42, help="Global RNG seed")
    p.add_argument("--max-days", type=int, default=0, help="Limit number of schedule days (0 = all)")
    p.add_argument("--max-players", type=int, default=0, help="Limit number of players (0 = all)")
    p.add_argument(
        "--skip-template-baseline",
        action="store_true",
        help="Skip reading the StatsSports example CSV for baseline medians (faster, uses built-in defaults)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    baseline = default_baseline() if args.skip_template_baseline else read_template_baseline()
    schedule = load_schedule()
    if args.max_days and args.max_days > 0:
        schedule = schedule[: args.max_days]

    players = load_players(args.team)
    if args.max_players and args.max_players > 0:
        players = players[: args.max_players]
    squad_size = len(players)

    rows: list[dict] = []

    for day in schedule:
        d = _parse_iso_date(day["date"])
        label = day.get("label", "")
        session_name = _safe_text(day.get("session_name") or "Training")
        contents = day.get("contents") or []
        if not contents:
            continue

        label_norm = str(label or "").strip().upper()

        # Convert planned distance (km) into a per-player anchor distance.
        # We want training-day targets to be a % of match-day load. Therefore we keep the
        # same effective denominator (11) across MD and training days so that the label
        # factor maps directly to per-player load ratios.
        team_km = float(day.get("planned_total_distance_gross_km") or 0.0)
        if team_km <= 0:
            # e.g., MD+2 "Off" has factor 0.0 -> don't emit pseudo-training with baseline distance.
            continue

        per_player_anchor_distance_m = (team_km * 1000.0) / 11.0

        # Anchor total duration from baseline, adjusted by distance ratio
        dist_ratio = per_player_anchor_distance_m / baseline.total_distance_m if baseline.total_distance_m > 0 else 1.0
        if label_norm == "MD":
            # Allow a full match block; activation + match should end up around 95–110 minutes.
            total_duration_s = _clamp(baseline.total_duration_s * (0.95 + 0.35 * dist_ratio), 80 * 60, 115 * 60)
        else:
            total_duration_s = _clamp(baseline.total_duration_s * (0.85 + 0.3 * dist_ratio), 30 * 60, 75 * 60)

        # Allocate duration and distance across contents
        dist_weights = []
        dur_weights = []
        profiles = []
        for c in contents:
            p = content_profile(c.get("type"))
            profiles.append(p)
            dist_weights.append(p["dist_w"])
            dur_weights.append(p["dur_w"])

        dist_shares = allocate(dist_weights)
        dur_shares = allocate(dur_weights)

        content_blocks = []
        for idx, c in enumerate(contents):
            dur_s = max(5 * 60, total_duration_s * dur_shares[idx])
            content_blocks.append(
                {
                    "type": c.get("type") or "",
                    "name": _safe_text(c.get("name") or f"Block {idx+1}"),
                    "duration_s": dur_s,
                    "distance_m": per_player_anchor_distance_m * dist_shares[idx],
                    "profile": profiles[idx],
                }
            )

        # Normalize again after min-duration enforcement
        sum_dur = sum(b["duration_s"] for b in content_blocks)
        if sum_dur > 0:
            scale = total_duration_s / sum_dur
            for b in content_blocks:
                b["duration_s"] *= scale

        # Time bookkeeping
        day_start = _epoch_seconds_local_noon(d) - (total_duration_s / 2.0)
        cursor = day_start

        activity_id = _uuid_for_activity(d, label)
        activity_name = f"{session_name} {label}".strip()

        for block in content_blocks:
            period_name = block["name"]
            if str(block.get("type") or "").lower() == "activation":
                period_name = standardized_activation_name(d, label, period_name)
            period_id = _uuid_for_period(d, label, period_name)
            dur_s = float(block["duration_s"])
            end = cursor + dur_s

            # content-level anchors
            mpm = (block["distance_m"] / (dur_s / 60.0)) if dur_s > 0 else baseline.meter_per_min
            # clamp to reasonable range
            mpm = _clamp(mpm, 50.0, 160.0)

            # speed zone distribution
            base = baseline.vel_share_1to5
            mult = block["profile"]["vel_mult"]
            alpha = [max(0.05, base[i] * mult[i] * 50.0) for i in range(5)]
            vel_shares = _dirichlet(alpha, rng)

            for player_name in players:
                player_rng = random.Random(uuid.uuid5(uuid.NAMESPACE_DNS, f"{d.isoformat()}::{label}::{period_name}::{player_name}").int & 0xFFFFFFFF)
                player_factor = _clamp(player_rng.normalvariate(1.0, 0.07), 0.85, 1.18)

                duration_s_player = dur_s
                total_distance = block["distance_m"] * player_factor

                meter_per_min = (total_distance / (duration_s_player / 60.0)) if duration_s_player > 0 else mpm

                # PlayerLoad approx from baseline relationship
                pl_per_min = _clamp(baseline.player_load_per_min * (meter_per_min / baseline.meter_per_min) * player_factor, 6.0, 16.5)
                total_player_load = pl_per_min * (duration_s_player / 60.0)
                peak_player_load = _clamp(baseline.peak_player_load * (1.0 + 0.25 * (block["profile"]["acc_mult"] - 1.0)), 3.0, 12.0)

                # Max speed: anchored, boosted by content profile
                max_vel = _clamp(baseline.max_vel * block["profile"]["maxv_mult"] * (0.95 + 0.1 * player_factor), 18.0, 36.0)

                # Velocity band distances
                vdist = [total_distance * vel_shares[i] for i in range(5)]

                # Acceleration zones 1–5 (template values are mostly zero, so we synthesize plausible counts/distance)
                minutes = duration_s_player / 60.0
                acc_intensity = block["profile"]["acc_mult"]
                dec_intensity = block["profile"]["dec_mult"]

                # baseline from decel_per_min (since accel bands are not populated in template)
                base_events_per_min = _clamp(baseline.decel_per_min * 0.9, 0.4, 2.2)

                total_acc_events = player_rng.poisson(lam=base_events_per_min * acc_intensity * minutes) if hasattr(player_rng, 'poisson') else int(player_rng.gammavariate(base_events_per_min * acc_intensity * max(1.0, minutes), 1.0))
                total_dec_events = player_rng.poisson(lam=baseline.decel_per_min * dec_intensity * minutes) if hasattr(player_rng, 'poisson') else int(player_rng.gammavariate(baseline.decel_per_min * dec_intensity * max(1.0, minutes), 1.0))

                # Split events into zones (heavier zones get more share in intense drills)
                acc_zone_alpha = [2.0, 2.5, 2.5, 2.0, 1.5]
                dec_zone_alpha = [2.2, 2.6, 2.6, 2.1, 1.6]
                if acc_intensity >= 1.3:
                    acc_zone_alpha = [1.4, 2.2, 2.8, 2.6, 2.2]
                if dec_intensity >= 1.4:
                    dec_zone_alpha = [1.4, 2.2, 2.9, 2.7, 2.3]

                acc_sh = _dirichlet(acc_zone_alpha, player_rng)
                dec_sh = _dirichlet(dec_zone_alpha, player_rng)

                acc_counts = [int(round(total_acc_events * acc_sh[i])) for i in range(5)]
                dec_counts = [int(round(total_dec_events * dec_sh[i])) for i in range(5)]

                # Fix rounding drift
                def _fix_counts(counts: list[int], target: int) -> list[int]:
                    diff = target - sum(counts)
                    if diff == 0:
                        return counts
                    idx = max(range(len(counts)), key=lambda i: counts[i])
                    counts[idx] = max(0, counts[idx] + diff)
                    return counts

                acc_counts = _fix_counts(acc_counts, total_acc_events)
                dec_counts = _fix_counts(dec_counts, total_dec_events)

                # Distance per accel/decel event (rough)
                acc_avg_dist = [0.8, 1.2, 1.6, 2.0, 2.4]
                dec_avg_dist = [0.7, 1.1, 1.5, 1.9, 2.3]
                acc_dist = [acc_counts[i] * acc_avg_dist[i] for i in range(5)]
                dec_dist = [dec_counts[i] * dec_avg_dist[i] for i in range(5)]

                # Metabolic power band distances – approximate by speed shares but shift towards higher bands in intense drills
                metab_mult = [1.0, 1.0, 1.0, 1.0, 1.0]
                if acc_intensity >= 1.3:
                    metab_mult = [0.8, 0.9, 1.1, 1.2, 1.3]
                metab_alpha = [max(0.05, vel_shares[i] * metab_mult[i] * 40.0) for i in range(5)]
                metab_sh = _dirichlet(metab_alpha, player_rng)
                metab_dist = [total_distance * metab_sh[i] for i in range(5)]

                # Decel summary metrics
                decel_per_minute = total_dec_events / minutes if minutes > 0 else 0.0
                max_effort_deceleration = float(baseline.max_effort_decel) * (0.9 + 0.2 * dec_intensity)  # negative

                row = {
                    "Activity_ID": activity_id,
                    "date": _format_statsports_date(d),
                    "date_iso": d.isoformat(),
                    "label": label,
                    "team_name": args.team,
                    "athlete_id": _uuid_for_player(player_name),
                    "athlete_name": player_name,
                    "activity_name": activity_name,
                    "period_id": period_id,
                    "period_name": period_name,
                    "content_type": block["type"],
                    "start_time": round(cursor, 2),
                    "end_time": round(end, 2),
                    "field_time": round(duration_s_player, 2),
                    "total_duration": round(duration_s_player, 2),
                    "total_distance": round(total_distance, 2),
                    "meterage_per_minute": round(meter_per_min, 2),
                    "total_player_load": round(total_player_load, 2),
                    "peak_player_load": round(peak_player_load, 2),
                    "player_load_per_minute": round(pl_per_min, 2),
                    "max_vel": round(max_vel, 2),
                    "max_effort_deceleration": round(max_effort_deceleration, 3),
                    "deceleration_per_minute": round(decel_per_minute, 3),

                    # required band distances
                    "velocity_band1_total_distance": round(vdist[0], 2),
                    "velocity_band2_total_distance": round(vdist[1], 2),
                    "velocity_band3_total_distance": round(vdist[2], 2),
                    "velocity_band4_total_distance": round(vdist[3], 2),
                    "velocity_band5_total_distance": round(vdist[4], 2),

                    "acceleration_band1_total_effort_count": acc_counts[0],
                    "acceleration_band2_total_effort_count": acc_counts[1],
                    "acceleration_band3_total_effort_count": acc_counts[2],
                    "acceleration_band4_total_effort_count": acc_counts[3],
                    "acceleration_band5_total_effort_count": acc_counts[4],

                    "acceleration_band1_total_distance": round(acc_dist[0], 2),
                    "acceleration_band2_total_distance": round(acc_dist[1], 2),
                    "acceleration_band3_total_distance": round(acc_dist[2], 2),
                    "acceleration_band4_total_distance": round(acc_dist[3], 2),
                    "acceleration_band5_total_distance": round(acc_dist[4], 2),

                    # custom derived decel zones 1–5
                    "deceleration_band1_total_effort_count": dec_counts[0],
                    "deceleration_band2_total_effort_count": dec_counts[1],
                    "deceleration_band3_total_effort_count": dec_counts[2],
                    "deceleration_band4_total_effort_count": dec_counts[3],
                    "deceleration_band5_total_effort_count": dec_counts[4],

                    "deceleration_band1_total_distance": round(dec_dist[0], 2),
                    "deceleration_band2_total_distance": round(dec_dist[1], 2),
                    "deceleration_band3_total_distance": round(dec_dist[2], 2),
                    "deceleration_band4_total_distance": round(dec_dist[3], 2),
                    "deceleration_band5_total_distance": round(dec_dist[4], 2),

                    # metabolic distances
                    "metabolic_power_band1_total_distance": round(metab_dist[0], 2),
                    "metabolic_power_band2_total_distance": round(metab_dist[1], 2),
                    "metabolic_power_band3_total_distance": round(metab_dist[2], 2),
                    "metabolic_power_band4_total_distance": round(metab_dist[3], 2),
                    "metabolic_power_band5_total_distance": round(metab_dist[4], 2),
                }

                rows.append(row)

            cursor = end

    # Write CSV (utf-8 with BOM for Excel friendliness)
    if not rows:
        raise RuntimeError("No rows generated")

    fieldnames = list(rows[0].keys())
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

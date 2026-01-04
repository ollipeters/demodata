import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEAM_FILE = ROOT / "dfl_running_team_matches.json"

rows = json.loads(TEAM_FILE.read_text(encoding="utf-8"))

print(f"File: {TEAM_FILE}")
print(f"Total rows: {len(rows)}")

key = lambda r: (str(r.get("match_id", "")), str(r.get("team", "")))

counts = Counter(key(r) for r in rows)
dup_keys = [(k, c) for k, c in counts.items() if c > 1]
print(f"Duplicate keys (match_id, team): {len(dup_keys)}")

if dup_keys:
    print("\nTop duplicates:")
    for (match_id, team), c in sorted(dup_keys, key=lambda x: (-x[1], x[0][0], x[0][1]))[:20]:
        print(f"- {(match_id, team)} x{c}")

    # Show detail for the top dup
    (match_id, team), c = sorted(dup_keys, key=lambda x: (-x[1], x[0][0], x[0][1]))[0]
    detail = [r for r in rows if key(r) == (match_id, team)]
    print("\nDetails for top duplicate:")
    for r in detail:
        print({
            "match_id": r.get("match_id"),
            "match_date": r.get("match_date"),
            "team": r.get("team"),
            "opponent": r.get("opponent"),
            "minutes": r.get("minutes"),
            "match_minutes_gross": r.get("match_minutes_gross"),
            "total_distance_gross_km": r.get("total_distance_gross_km"),
            "intense_distance_gross_km": r.get("intense_distance_gross_km"),
            "source_file": r.get("source_file"),
        })

# Specific Freiburg <-> Leverkusen focus

def has(s: str, needle: str) -> bool:
    return needle in (s or "")

pair_rows = [
    r
    for r in rows
    if (
        (has(str(r.get("team", "")), "Freiburg") and has(str(r.get("opponent", "")), "Leverkusen"))
        or (has(str(r.get("team", "")), "Leverkusen") and has(str(r.get("opponent", "")), "Freiburg"))
    )
]
print(f"\nFreiburg<->Leverkusen rows: {len(pair_rows)}")

if pair_rows:
    print("Freiburg<->Leverkusen row details:")
    for r in sorted(pair_rows, key=lambda r: (str(r.get('match_date','')), str(r.get('team','')))):
        print({
            "match_id": r.get("match_id"),
            "match_date": r.get("match_date"),
            "team": r.get("team"),
            "opponent": r.get("opponent"),
        })

pair_counts = Counter(key(r) for r in pair_rows)
pair_dups = [(k, c) for k, c in pair_counts.items() if c > 1]
print(f"Freiburg<->Leverkusen duplicate keys (match_id, team): {len(pair_dups)}")
for (match_id, team), c in sorted(pair_dups, key=lambda x: (-x[1], x[0][0], x[0][1]))[:20]:
    print(f"- {(match_id, team)} x{c}")

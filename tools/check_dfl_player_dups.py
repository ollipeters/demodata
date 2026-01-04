import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAYER_FILE = ROOT / "dfl_running_players.json"

rows = json.loads(PLAYER_FILE.read_text(encoding="utf-8"))

print(f"File: {PLAYER_FILE}")
print(f"Total rows: {len(rows)}")

key = lambda r: (str(r.get("match_id", "")), str(r.get("team", "")), str(r.get("player_name", "")))

counts = Counter(key(r) for r in rows)
dup_keys = [(k, c) for k, c in counts.items() if c > 1]
print(f"Duplicate keys (match_id, team, player_name): {len(dup_keys)}")

if dup_keys:
    print("Top duplicates:")
    for (mid, team, player), c in sorted(dup_keys, key=lambda x: (-x[1], x[0][0], x[0][1], x[0][2]))[:30]:
        print(f"- {(mid, team, player)} x{c}")

# Focus: Freiburg vs Leverkusen match id (from team-matches checker)
match_id = "DFL-MAT-J041VX"
focus = [r for r in rows if str(r.get('match_id','')) == match_id]
print(f"\nRows for match_id={match_id}: {len(focus)}")

focus_counts = Counter(key(r) for r in focus)
focus_dups = [(k, c) for k, c in focus_counts.items() if c > 1]
print(f"Duplicates within match_id={match_id}: {len(focus_dups)}")
if focus_dups:
    for (mid, team, player), c in sorted(focus_dups, key=lambda x: (-x[1], x[0][1], x[0][2]))[:50]:
        print(f"- {(team, player)} x{c}")

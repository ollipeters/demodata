import json
from collections import Counter
from pathlib import Path


def iter_json_array_objects(path: Path, limit: int = 200):
    """Stream objects from a top-level JSON array without loading the full file."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        buf = ""
        # Read until we hit '['
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                raise RuntimeError("Unexpected EOF while searching for '['")
            buf += chunk
            i = buf.find("[")
            if i != -1:
                buf = buf[i + 1 :]
                break

        count = 0
        while count < limit:
            # Skip whitespace/commas
            j = 0
            while j < len(buf) and buf[j] in " \t\r\n,":
                j += 1
            buf = buf[j:]

            # End of array?
            if buf.startswith("]"):
                return

            # Ensure we have enough buffer to decode a full object
            while True:
                try:
                    obj, idx = decoder.raw_decode(buf)
                    buf = buf[idx:]
                    yield obj
                    count += 1
                    break
                except json.JSONDecodeError:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        raise
                    buf += chunk


def main():
    path = Path(__file__).resolve().parents[1] / "events_data.json"
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    key_counts = Counter()
    event_type_counts = Counter()

    shot_key_counts = Counter()
    shot_xg_samples = []

    card_type_counts = Counter()

    for ev in iter_json_array_objects(path, limit=500):
        if isinstance(ev, dict):
            for k in ev.keys():
                key_counts[k] += 1

            et = ev.get("event_type")
            if et is not None:
                event_type_counts[str(et)] += 1

            if ev.get("isShot") is True or et in {"Shot", "MissedShots", "SavedShot", "Goal"}:
                for k in ev.keys():
                    shot_key_counts[k] += 1

                # Try common xG field names
                for xg_key in ("xg", "xG", "expectedGoals", "expected_goals", "shot_xg"):
                    if xg_key in ev and ev.get(xg_key) is not None:
                        shot_xg_samples.append((xg_key, ev.get(xg_key)))
                        break

            if et in {"Card", "RedCard", "YellowCard", "SecondYellow"}:
                for k in ("cardType", "card_type", "type", "outcome"):
                    if k in ev and ev.get(k) is not None:
                        card_type_counts[str(ev.get(k))] += 1

    print("File:", path)
    print("\nTop-level event keys (sampled):")
    for k, c in key_counts.most_common():
        print(f"  {k}: {c}")

    print("\nEvent types (sampled):")
    for k, c in event_type_counts.most_common(30):
        print(f"  {k}: {c}")

    print("\nShot-event keys (sampled):")
    for k, c in shot_key_counts.most_common():
        print(f"  {k}: {c}")

    print("\nDetected xG samples (first 20):")
    for k, v in shot_xg_samples[:20]:
        print(f"  {k} = {v}")

    if card_type_counts:
        print("\nCard type samples:")
        for k, c in card_type_counts.most_common(20):
            print(f"  {k}: {c}")


if __name__ == "__main__":
    main()

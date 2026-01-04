import csv
import glob
import json
import os
from pathlib import Path


def _safe_int(v):
    try:
        return int(float(v))
    except Exception:
        return None


def main():
    root = Path(__file__).resolve().parent
    data_dir = root / 'Data'
    whoscored_dir = data_dir / 'whoscored'

    formation_files = sorted(glob.glob(str(data_dir / '*_formations.csv')))
    all_rows = []

    def _add_row(match_id, row):
        team_name = (row.get('team_name') or '').strip() or None
        formation_name = (row.get('formationName') or '').strip() or None

        player_ids_raw = row.get('playerIds')
        slots_raw = row.get('formationSlots')
        try:
            player_ids = json.loads(player_ids_raw) if isinstance(player_ids_raw, str) and player_ids_raw else player_ids_raw
        except Exception:
            player_ids = None
        try:
            formation_slots = json.loads(slots_raw) if isinstance(slots_raw, str) and slots_raw else slots_raw
        except Exception:
            formation_slots = None

        all_rows.append({
            'match_id': match_id,
            'team_name': team_name,
            'formationName': formation_name,
            'field': (row.get('field') or '').strip() or None,
            'period': _safe_int(row.get('period')),
            'startMinuteExpanded': _safe_int(row.get('startMinuteExpanded')),
            'endMinuteExpanded': _safe_int(row.get('endMinuteExpanded')),
            'captainPlayerName': (row.get('captainPlayerName') or '').strip() or None,
            'playerIds': player_ids,
            'formationSlots': formation_slots,
        })

    for fp in formation_files:
        p = Path(fp)
        match_id = p.name.replace('_formations.csv', '')
        with p.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                _add_row(match_id, row)

    # Also load formations from WhoScored match JSONs.
    # This is the main source for seasons where *_formations.csv isn't available for all matches.
    if whoscored_dir.exists():
        json_files = sorted(whoscored_dir.glob('**/*.json'))
        for p in json_files:
            # Skip index.json or other non-match files by structure check.
            try:
                payload = json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                continue

            parsed = payload.get('parsed') if isinstance(payload, dict) else None
            formations = parsed.get('formations') if isinstance(parsed, dict) else None
            if not isinstance(formations, list) or not formations:
                continue

            match_id = p.stem
            for fr in formations:
                if not isinstance(fr, dict):
                    continue
                _add_row(match_id, fr)

    # De-duplicate exact duplicates (can happen if a match exists as CSV and JSON).
    # Keep formation segments distinct (start/end minute is part of the key).
    uniq = []
    seen = set()
    for r in all_rows:
        k = (
            str(r.get('match_id') or ''),
            str(r.get('team_name') or ''),
            str(r.get('formationName') or ''),
            str(r.get('field') or ''),
            str(r.get('period') or ''),
            str(r.get('startMinuteExpanded') or ''),
            str(r.get('endMinuteExpanded') or ''),
            str(r.get('captainPlayerName') or ''),
        )
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    all_rows = uniq

    out_path = root / 'formations_data.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(all_rows, f, ensure_ascii=False)

    print(f'Wrote {len(all_rows)} rows -> {out_path}')


if __name__ == '__main__':
    main()

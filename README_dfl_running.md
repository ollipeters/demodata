# DFL Running Data Prep

This workspace contains DFL xlsx exports in `Data/dfl/` (sheet: `Spieler Statistiken`).

## What the script does

- Reads all `*.xlsx` files in `Data/dfl/`
- Filters player rows (minutes > 0 when available)
- Aggregates the selected `--team` to **team totals per match**
- Keeps match identifiers so **team + date** are always traceable

## Outputs

- `dfl_running_team_matches.json` (per match, team totals)
- `dfl_running_team_matches.csv`

Optional:
- `dfl_running_players.json` and `.csv` (per player per match)

## Run

```powershell
# Use ALL to include both teams per match.
# This is required for Gegneranalyse -> Physisch to work for opponents that haven't played HSV yet.
C:/Users/ollip/Desktop/DataAnalystHSV/.venv/Scripts/python.exe prepare_dfl_running_data.py --team ALL --input-dir "Data/dfl" --out-dir "." --write-player-json
```

If you need dependencies in a fresh venv:

```powershell
C:/Users/ollip/Desktop/DataAnalystHSV/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

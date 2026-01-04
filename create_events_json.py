import pandas as pd
import glob
import json
import os
import numpy as np

def create_events_json():
    """
    Erstellt eine einzige events_data.json Datei.

    Priorität:
    1) WhoScored Match-JSONs unter Data/whoscored/<team>/*.json (parsed.events)
    2) Fallback: Data/*_events.csv
    """
    print("Lade Event-Daten...")

    # --- 1) WhoScored JSONs ---
    ws_files = [
        p for p in glob.glob(os.path.join("Data", "whoscored", "*", "*.json"))
        if not p.endswith(os.path.join("whoscored", "index.json")) and not p.endswith("index.json")
    ]

    if ws_files:
        print(f"Nutze WhoScored JSONs: {len(ws_files)} Dateien")
        rows = []
        for path in sorted(ws_files):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
            except Exception:
                continue

            status_ok = (payload.get('status') or {}).get('ok', True)
            if status_ok is not True:
                continue

            match_id = os.path.splitext(os.path.basename(path))[0]
            parsed = payload.get('parsed') or {}
            events = parsed.get('events') or []
            if not isinstance(events, list) or not events:
                continue

            for e in events:
                if not isinstance(e, dict):
                    continue
                r = dict(e)
                r['match_id'] = match_id
                rows.append(r)

        if not rows:
            print("Keine Events in WhoScored JSONs gefunden.")
            return

        combined_events_df = pd.DataFrame(rows)

    else:
        # --- 2) Fallback: CSVs ---
        event_files = glob.glob("Data/*_events.csv")
        if not event_files:
            print("Keine WhoScored JSONs und keine *_events.csv Dateien gefunden.")
            return

        all_events = []
        for file in event_files:
            match_id = os.path.basename(file).replace('_events.csv', '')
            df = pd.read_csv(file)
            df['match_id'] = match_id
            all_events.append(df)

        if not all_events:
            print("Keine Events zum Verarbeiten.")
            return

        combined_events_df = pd.concat(all_events, ignore_index=True)

    # Die Spalte 'is_goalkick' wird jetzt direkt vom Scraper geliefert.
    # Wir stellen nur sicher, dass sie existiert und füllen fehlende Werte mit False.
    if 'is_goalkick' not in combined_events_df.columns:
        print("Warnung: 'is_goalkick' Spalte nicht gefunden. Wird als False initialisiert.")
        combined_events_df['is_goalkick'] = False
    combined_events_df['is_goalkick'] = combined_events_df['is_goalkick'].fillna(False)
    # robust to "True"/"False" strings
    combined_events_df['is_goalkick'] = combined_events_df['is_goalkick'].apply(lambda v: str(v).strip().lower() in ['true','1','t','yes','y'])

    # --- NEU: Berechnung der durchschnittlichen Abstossdistanz ---
    goalkicks = combined_events_df[combined_events_df['is_goalkick'] == True].copy()
    if not goalkicks.empty:
        # Ensure numeric coords exist
        for col in ['x_coord','y_coord','end_x','end_y']:
            if col in goalkicks.columns:
                goalkicks[col] = pd.to_numeric(goalkicks[col], errors='coerce')
        # Distanz in Metern berechnen (Feldlänge 105m, Breite 68m)
        if all(c in goalkicks.columns for c in ['x_coord','y_coord','end_x','end_y']):
            goalkicks = goalkicks.dropna(subset=['x_coord','y_coord','end_x','end_y'])
            goalkicks['distance_m'] = np.sqrt(
                ((goalkicks['end_x'] - goalkicks['x_coord']) * 1.05)**2 +
                ((goalkicks['end_y'] - goalkicks['y_coord']) * 0.68)**2
            )
        else:
            goalkicks['distance_m'] = np.nan
        
        # Durchschnitt pro Team und Spiel berechnen
        if 'team' in goalkicks.columns:
            avg_distance = goalkicks.groupby(['match_id', 'team'])['distance_m'].mean().round(0).reset_index(name='avg_goalkick_distance')
        else:
            avg_distance = pd.DataFrame(columns=['match_id','team','avg_goalkick_distance'])

        # Den berechneten Wert zurück in den Haupt-DataFrame mergen
        # Wir mergen es auf alle Events eines Teams in einem Spiel, damit es leicht zugänglich ist
        combined_events_df = pd.merge(
            combined_events_df,
            avg_distance,
            on=['match_id', 'team'],
            how='left'
        )
        if 'avg_goalkick_distance' in combined_events_df.columns:
            combined_events_df['avg_goalkick_distance'] = pd.to_numeric(combined_events_df['avg_goalkick_distance'], errors='coerce').fillna(0)
    
    with open('events_data.json', 'w', encoding='utf-8') as f:
        combined_events_df.to_json(f, orient='records', indent=2, force_ascii=False)
    
    print(f"FERTIG! events_data.json mit {len(combined_events_df)} Events wurde erstellt.")

if __name__ == "__main__":
    create_events_json()
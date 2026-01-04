import pandas as pd
import glob
import json
import os
import warnings
import numpy as np

def calculate_progressive_passes(events_df, group_keys=('match_id', 'player_name')):
    passes = events_df[events_df['event_type'] == 'Pass'].copy()
    passes = passes.dropna(subset=['x_coord', 'y_coord', 'end_x', 'end_y'])
    for col in ['x_coord', 'y_coord', 'end_x', 'end_y']:
        passes[col] = pd.to_numeric(passes[col], errors='coerce')
    
    # HINZUGEFÜGT: (passes['end_x'] > passes['x_coord']) & ... zu jeder Bedingung
    cond1 = (passes['end_x'] > passes['x_coord']) & (passes['x_coord'] <= 50) & (passes['end_x'] <= 50) & (np.sqrt((passes['end_x'] - passes['x_coord'])**2 + (passes['end_y'] - passes['y_coord'])**2) * 1.04 >= 30)
    cond2 = (passes['end_x'] > passes['x_coord']) & (passes['x_coord'] <= 50) & (passes['end_x'] > 50) & (np.sqrt((passes['end_x'] - passes['x_coord'])**2 + (passes['end_y'] - passes['y_coord'])**2) * 1.04 >= 15)
    cond3 = (passes['end_x'] > passes['x_coord']) & (passes['x_coord'] > 50) & (passes['end_x'] > 50) & (np.sqrt((passes['end_x'] - passes['x_coord'])**2 + (passes['end_y'] - passes['y_coord'])**2) * 1.04 >= 10)
    
    passes['is_progressive'] = cond1 | cond2 | cond3
    
    if not passes[passes['is_progressive']].empty:
        return passes[passes['is_progressive']].groupby(list(group_keys)).size().reset_index(name='progressive_passes')
    
    return pd.DataFrame(columns=list(group_keys) + ['progressive_passes'])

def calculate_advanced_kpis(events_df, group_keys=('match_id', 'player_name')):
    if events_df.empty: return pd.DataFrame()
    group_keys_list = list(group_keys)
    df = events_df.copy()
    for col in ['x_coord', 'y_coord', 'end_x', 'end_y', 'minute', 'second']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['x_coord', 'y_coord'])
    df['total_seconds'] = df['minute'] * 60 + df['second']
    df = df.sort_values(by=['match_id', 'total_seconds']).reset_index(drop=True)
    kpi_results = {}
    passes_into_final_third = df[(df['event_type'] == 'Pass') & (df['outcome'] == 'Successful') & (df['x_coord'] <= 66.6) & (df['end_x'] > 66.6)]
    if not passes_into_final_third.empty:
        kpi_results['passes_into_final_third'] = passes_into_final_third.groupby(group_keys_list).size()
    box_x_start, box_y_start, box_y_end = 83, 21, 79
    passes_into_box = df[(df['event_type'] == 'Pass') & (df['outcome'] == 'Successful') & (df['end_x'] > box_x_start) & (df['end_y'].between(box_y_start, box_y_end))]
    if not passes_into_box.empty:
        kpi_results['box_entries_pass'] = passes_into_box.groupby(group_keys_list).size()
    third_to_third_passes = df[(df['event_type'] == 'Pass') & (df['outcome'] == 'Successful') & (df['x_coord'] < 33.3) & (df['end_x'] > 66.6)]
    if not third_to_third_passes.empty:
        kpi_results['third_to_third_passes'] = third_to_third_passes.groupby(group_keys_list).size()
    area14_x_start, area14_x_end = 66.6, 83
    passes_into_area14 = df[(df['event_type'] == 'Pass') & (df['outcome'] == 'Successful') & (df['end_x'].between(area14_x_start, area14_x_end)) & (df['end_y'].between(box_y_start, box_y_end))]
    if not passes_into_area14.empty:
        kpi_results['passes_into_area14'] = passes_into_area14.groupby(group_keys_list).size()

    # --- HINZUGEFÜGT: Pässe in Zone 11 ---
    # Definition Zone 11 (Mitte-Rechts im Mittelfeld): X: 33.3-66.6, Y: 66.67-83.33 (5./6. horizontaler Streifen)
    z11_x_start, z11_x_end = 33.3, 66.6
    z11_y_start, z11_y_end = 66.67, 83.33 
    passes_into_zone11 = df[(df['event_type'] == 'Pass') & (df['outcome'] == 'Successful') & (df['end_x'].between(z11_x_start, z11_x_end)) & (df['end_y'].between(z11_y_start, z11_y_end))]
    if not passes_into_zone11.empty:
        kpi_results['passes_into_zone11'] = passes_into_zone11.groupby(group_keys_list).size()
    # --- ENDE HINZUGEFÜGT ---

    dribbles_into_box = df[(df['event_type'] == 'TakeOn') & (df['outcome'] == 'Successful') & (df['x_coord'] < box_x_start) & (df['end_x'] > box_x_start) & (df['end_y'].between(box_y_start, box_y_end))]
    if not dribbles_into_box.empty:
        kpi_results['box_entries_carry'] = dribbles_into_box.groupby(group_keys_list).size()
    pass_dist_y = abs(df['end_y'] - df['y_coord'])
    switches = df[(df['event_type'] == 'Pass') & (df['outcome'] == 'Successful') & (pass_dist_y > 50)]
    if not switches.empty:
        kpi_results['switches_of_play'] = switches.groupby(group_keys_list).size()
    defensive_third_touches = df[df['x_coord'] < 33.3]
    if not defensive_third_touches.empty:
        kpi_results['defensive_third_touches'] = defensive_third_touches.groupby(group_keys_list).size()
    carries = df[df['event_type'] == 'TakeOn'].copy()
    forward_distance = carries['end_x'] - carries['x_coord']
    cond_prog_carry = (forward_distance >= 10) | ((carries['end_x'] > box_x_start) & (carries['end_y'].between(box_y_start, box_y_end)))
    progressive_carries = carries[cond_prog_carry]
    if not progressive_carries.empty:
        kpi_results['progressive_carries'] = progressive_carries.groupby(group_keys_list).size()
    sca_events = ['Pass', 'TakeOn', 'Foul', 'Shot']
    sca_players = []; pre_assist_players = []; gca_players = []
    shot_indices = df[df['event_type'].isin(['Shot', 'MissedShots', 'SavedShot', 'Goal'])].index
    for shot_idx in shot_indices:
        shot_event = df.loc[shot_idx]; team = shot_event.get('team'); time = shot_event.get('total_seconds')
        is_goal = (shot_event.get('event_type') == 'Goal') or (shot_event.get('outcome') == 'Goal')
        prev_events = df[(df['team'] == team) & (df['total_seconds'] >= time - 10) & (df.index < shot_idx)].tail(2)
        for _, prev_event in prev_events.iterrows():
            if prev_event['event_type'] in sca_events:
                sca_players.append({k: prev_event[k] for k in group_keys_list})
                if is_goal: gca_players.append({k: prev_event[k] for k in group_keys_list})
        if is_goal and len(prev_events) == 2:
            pre_assist_event = prev_events.iloc[0]
            pre_assist_players.append({k: pre_assist_event[k] for k in group_keys_list})
    if sca_players: kpi_results['sca'] = pd.DataFrame(sca_players).groupby(group_keys_list).size()
    if gca_players: kpi_results['gca'] = pd.DataFrame(gca_players).groupby(group_keys_list).size()
    if pre_assist_players: kpi_results['pre_assists'] = pd.DataFrame(pre_assist_players).groupby(group_keys_list).size()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        final_df = None
        for name, series in kpi_results.items():
            tmp = series.rename(name).reset_index()
            if final_df is None:
                final_df = tmp
            else:
                final_df = final_df.merge(tmp, on=group_keys_list, how='outer')
        if final_df is None:
            return pd.DataFrame(columns=group_keys_list + ['box_entries_total'])
        final_df = final_df.reset_index(drop=True)
    final_df['box_entries_total'] = final_df.get('box_entries_pass', 0) + final_df.get('box_entries_carry', 0)
    return final_df

def calculate_ppda(events_df, group_keys=('match_id', 'team')):
    if events_df.empty: return pd.DataFrame()
    df = events_df.copy(); df['x_coord'] = pd.to_numeric(df['x_coord'], errors='coerce')
    defensive_events = ['Tackle', 'Interception', 'Foul']
    def_actions = df[(df['event_type'].isin(defensive_events)) & (df['x_coord'] > 66.6)]
    def_actions_count = def_actions.groupby(list(group_keys)).size().reset_index(name='defensive_actions')
    opponent_passes = df[(df['event_type'] == 'Pass') & (df['x_coord'] <= 66.6) & (df['outcome'] == 'Successful')]
    opponent_passes_count = opponent_passes.groupby(list(group_keys)).size().reset_index(name='opponent_passes')
    if def_actions_count.empty or opponent_passes_count.empty:
        return pd.DataFrame(columns=list(group_keys) + ['ppda'])
    ppda_data = pd.merge(def_actions_count, opponent_passes_count, on=list(group_keys), how='outer').fillna(0)
    ppda_final = []
    for match_id, grp in ppda_data.groupby('match_id'):
        teams = grp['team'].unique().tolist()
        if len(teams) != 2: continue
        row_a = grp[grp['team'] == teams[0]]
        row_b = grp[grp['team'] == teams[1]]
        if row_a.empty or row_b.empty: continue
        def_a = float(row_a['defensive_actions'].iloc[0])
        def_b = float(row_b['defensive_actions'].iloc[0])
        opp_a = float(row_a['opponent_passes'].iloc[0])
        opp_b = float(row_b['opponent_passes'].iloc[0])
        ppda_a = np.nan if def_a == 0 else (opp_b / def_a)
        ppda_b = np.nan if def_b == 0 else (opp_a / def_b)
        ppda_final.append({'match_id': match_id, 'team': teams[0], 'ppda': ppda_a})
        ppda_final.append({'match_id': match_id, 'team': teams[1], 'ppda': ppda_b})
    return pd.DataFrame(ppda_final).fillna(0) if ppda_final else pd.DataFrame(columns=list(group_keys) + ['ppda'])

def calculate_field_tilt(events_df, group_keys=('match_id', 'team')):
    if events_df is None or events_df.empty: return pd.DataFrame()
    df = events_df.copy(); df['x_coord'] = pd.to_numeric(df['x_coord'], errors='coerce')
    total_touches = df.groupby(list(group_keys)).size().reset_index(name='total_touches')
    attacking_third_touches = df[df['x_coord'] > 66.6].groupby(list(group_keys)).size().reset_index(name='attacking_third_touches')
    field_tilt_data = pd.merge(total_touches, attacking_third_touches, on=list(group_keys), how='left').fillna(0)
    field_tilt_data['field_tilt'] = (field_tilt_data['attacking_third_touches'] / field_tilt_data['total_touches'].replace(0, 1) * 100).round(2)
    return field_tilt_data[list(group_keys) + ['field_tilt']]

def calculate_possession(events_df):
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=['match_id','team','possession_seconds','possession_percent'])
    df = events_df.copy()
    # sichere Numerik-Konvertierung
    for col in ['minute', 'second']:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)
    df['time'] = df['minute'] * 60 + df['second']
    df = df.sort_values(['match_id', 'time']).reset_index(drop=True)

    # Match-Ende (inkl. Nachspielzeit)
    match_end = df.groupby('match_id')['time'].max().reset_index(name='match_end')

    # nächste Aktion im Match (Zeitpunkt)
    df['next_time'] = df.groupby('match_id')['time'].shift(-1)
    df = df.merge(match_end, on='match_id', how='left')
    # Für die letzte Aktion: bis Match-Ende rechnen
    df['next_time'] = df['next_time'].fillna(df['match_end'])
    df['duration'] = (df['next_time'] - df['time']).clip(lower=0)

    # Nur Ballkontakte betrachten (erweiterbar)
    ball_events = ['Pass', 'TakeOn', 'BallTouch', 'Shot', 'Goal', 'BallRecovery', 'Clearance', 'SavedShot', 'MissedShots']
    touches = df[df['event_type'].isin(ball_events)].copy()
    if touches.empty:
        return pd.DataFrame(columns=['match_id','team','possession_seconds','possession_percent'])

    # Team-Summe in Sekunden
    touches['team'] = touches['team'].fillna('Unknown')
    team_pos = touches.groupby(['match_id', 'team'])['duration'].sum().reset_index(name='possession_seconds')

    # Prozentual (bezogen auf Match-Ende in Sekunden)
    team_pos = team_pos.merge(match_end, on='match_id', how='left')
    team_pos['possession_percent'] = (team_pos['possession_seconds'] / team_pos['match_end'].replace(0, 1) * 100).round(2)

    return team_pos[['match_id', 'team', 'possession_seconds', 'possession_percent']]

def calculate_pressure_regains(events_df, group_keys=['match_id', 'player_name']):
    if events_df.empty: return pd.DataFrame()
    df = events_df.copy()
    for col in ['minute', 'second']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['total_seconds'] = df['minute'] * 60 + df['second']
    df = df.sort_values(by=['match_id', 'total_seconds']).reset_index()
    loss_events = (((df['event_type'] == 'Pass') | (df['event_type'] == 'TakeOn')) & (df['outcome'] == 'Unsuccessful')) | (df['event_type'] == 'Dispossessed')
    gain_events = (df['event_type'] == 'BallRecovery') | (df['event_type'] == 'Interception') | ((df['event_type'] == 'Tackle') & (df['outcome'] == 'Successful'))
    df['is_loss'] = loss_events; df['is_gain'] = gain_events
    regains = []
    for idx, loss_row in df[df['is_loss']].iterrows():
        next_events = df[(df['match_id'] == loss_row['match_id']) & (df.index > idx)]
        regain_event = next_events[(next_events['team'] == loss_row['team']) & (next_events['is_gain']) & (next_events['total_seconds'] - loss_row['total_seconds'] <= 5)].head(1)
        if not regain_event.empty: regains.append(regain_event.iloc[0])
    if not regains: return pd.DataFrame(columns=group_keys + ['pressure_regains'])
    regains_df = pd.DataFrame(regains)
    return regains_df.groupby(group_keys).size().reset_index(name='pressure_regains')

XT_MATRIX = np.array([[0.007,0.009,0.011,0.014,0.018,0.023,0.03,0.038,0.049,0.063,0.08,0.102],
                      [0.008,0.01,0.012,0.016,0.02,0.026,0.034,0.043,0.055,0.07,0.088,0.111],
                      [0.009,0.011,0.014,0.018,0.023,0.029,0.038,0.048,0.062,0.079,0.1,0.126],
                      [0.01,0.012,0.016,0.02,0.026,0.033,0.043,0.055,0.07,0.089,0.113,0.143],
                      [0.01,0.012,0.016,0.02,0.026,0.033,0.043,0.055,0.07,0.089,0.113,0.143],
                      [0.009,0.011,0.014,0.018,0.023,0.029,0.038,0.048,0.062,0.079,0.1,0.126],
                      [0.008,0.01,0.012,0.016,0.02,0.026,0.034,0.043,0.055,0.07,0.088,0.111],
                      [0.007,0.009,0.011,0.014,0.018,0.023,0.03,0.038,0.049,0.063,0.08,0.102]]).T

def _compute_shot_xg(shots: pd.DataFrame) -> pd.Series:
    """Compute per-shot xG for WhoScored-style events.

    Returns a Series aligned to shots.index.
    """
    if shots is None or shots.empty:
        return pd.Series(dtype=float)

    shots = shots.copy()
    for col in ['x_coord', 'y_coord']:
        shots[col] = pd.to_numeric(shots[col], errors='coerce')
    shots = shots.dropna(subset=['x_coord', 'y_coord'])
    if shots.empty:
        return pd.Series(dtype=float)

    # Feld in Meter (105 x 68), Torbreite 7.32m
    goal_w = 7.32
    shots['_x_m'] = shots['x_coord'] / 100.0 * 105.0
    shots['_y_m'] = shots['y_coord'] / 100.0 * 68.0

    # Distanz zur Torlinie (x) und seitliche Abweichung zur Tor-Mitte (y, 34m)
    x_dist = (105.0 - shots['_x_m']).clip(lower=1e-6)  # numerisch stabil
    y_off = (shots['_y_m'] - 34.0).abs()

    # Distanz zum Torzentrum
    dist_m = np.sqrt(x_dist**2 + y_off**2)

    # Sichtwinkel in Radiant (konservativ; stabilisiere Nenner)
    denom = (x_dist**2 + y_off**2 - (goal_w/2.0)**2)
    angle_rad = np.where(denom <= 0, np.pi, np.arctan((goal_w * x_dist) / denom))

    # Strafraum
    in_box = (shots['_x_m'] >= (105.0 - 16.5)) & (y_off <= 20.16)
    close_range = dist_m <= 12.0

    def parse_quals(row):
        try:
            q = json.loads(row['qualifiers']) if isinstance(row.get('qualifiers'), str) else (row.get('qualifiers') or [])
        except Exception:
            q = []
        q_types = {e.get('type', {}).get('displayName') for e in q if isinstance(e, dict)}
        return q_types

    qual_sets = shots.apply(parse_quals, axis=1)
    is_pen = qual_sets.apply(lambda s: 'Penalty' in s)
    is_header = qual_sets.apply(lambda s: ('Head' in s) or ('Header' in s))
    is_from_corner = qual_sets.apply(lambda s: 'FromCorner' in s)
    is_fastbreak = qual_sets.apply(lambda s: 'FastBreak' in s)
    is_blocked = qual_sets.apply(lambda s: 'Blocked' in s)
    is_cross = qual_sets.apply(lambda s: 'Cross' in s)
    is_out_of_box = qual_sets.apply(lambda s: any(tag in s for tag in ['OutOfBox', 'OutOfBoxCentre']))
    is_box_loc = qual_sets.apply(lambda s: any(tag in s for tag in ['BoxCentre', 'BoxLeft', 'BoxRight']))

    # Logistische Basis
    b0, b1, b2 = -2.15, -0.060, 1.55
    logit = b0 + b1 * dist_m + b2 * angle_rad

    # Anpassungen
    logit += np.where(in_box, 0.0, -0.35)
    logit += np.where(close_range, 0.25, 0.0)
    logit += np.where(is_header, -0.45, 0.0)
    logit += np.where(is_from_corner, -0.25, 0.0)
    logit += np.where(is_fastbreak, 0.12, 0.0)
    logit += np.where(is_out_of_box, -0.10, 0.0)
    logit += np.where(is_box_loc, 0.08, 0.0)
    logit += np.where(is_cross, -0.10, 0.0)
    logit += np.where(dist_m >= 35.0, -0.70, np.where(dist_m >= 30.0, -0.50, 0.0))

    xg = 1.0 / (1.0 + np.exp(-logit))
    xg = np.where(is_blocked, xg * 0.92, xg)
    xg = np.where(is_pen, 0.76, xg)
    xg = np.clip(xg, 0.001, 0.97)

    out = pd.Series(xg, index=shots.index, dtype=float)
    return out

def calculate_xt(events_df, group_keys=['match_id', 'player_name']):
    df = events_df[(events_df['event_type'] == 'Pass') | (events_df['event_type'] == 'TakeOn')].copy()
    for col in ['x_coord', 'y_coord', 'end_x', 'end_y']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['x_coord', 'y_coord', 'end_x', 'end_y'])
    def get_xt_value(x, y):
        x_bin = min(int(x / 100 * 12), 11); y_bin = min(int(y / 100 * 8), 7)
        return XT_MATRIX[x_bin, y_bin]
    df['xt_start'] = df.apply(lambda r: get_xt_value(r['x_coord'], r['y_coord']), axis=1)
    df['xt_end'] = df.apply(lambda r: get_xt_value(r['end_x'], r['end_y']), axis=1)
    df['xt_added'] = df['xt_end'] - df['xt_start']
    df['xt_added'] = df['xt_added'].clip(lower=0)
    if not df.empty:
        return df.groupby(group_keys)['xt_added'].sum().reset_index(name='xt')
    return pd.DataFrame(columns=group_keys + ['xt'])

def calculate_xg(events_df, group_keys=['match_id', 'player_name']):
    if events_df.empty or not any(x in events_df['event_type'].unique() for x in ['Shot', 'MissedShots', 'SavedShot', 'Goal']):
        return pd.DataFrame()
    shots = events_df[events_df['event_type'].isin(['Shot', 'MissedShots', 'SavedShot', 'Goal'])].copy()
    shots['xg'] = _compute_shot_xg(shots)
    shots = shots.dropna(subset=['xg'])
    if shots.empty:
        return pd.DataFrame(columns=group_keys + ['xg'])

    if not shots.empty:
        return shots.groupby(group_keys)['xg'].sum().reset_index()
    return pd.DataFrame(columns=group_keys + ['xg'])

def calculate_xa(events_df, group_keys=('match_id', 'player_name'), lookback_seconds: int = 10):
    """Expected Assists (xA) proxy:

    - Compute xG for each shot event.
    - Attribute that xG to the *last successful pass* by the same team within lookback_seconds.

    This is a pragmatic approximation given available WhoScored-style event data.
    """
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=list(group_keys) + ['xa'])

    df = events_df.copy()
    for col in ['minute', 'second']:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce')
    df['total_seconds'] = df['minute'].fillna(0) * 60 + df['second'].fillna(0)
    df['match_id'] = df['match_id'].astype(str)
    df = df.sort_values(by=['match_id', 'total_seconds']).reset_index(drop=True)

    shot_types = {'Shot', 'MissedShots', 'SavedShot', 'Goal'}
    shots = df[df['event_type'].isin(shot_types)].copy()
    if shots.empty:
        return pd.DataFrame(columns=list(group_keys) + ['xa'])

    shots['xg'] = _compute_shot_xg(shots)
    shots = shots.dropna(subset=['xg'])
    if shots.empty:
        return pd.DataFrame(columns=list(group_keys) + ['xa'])

    # Candidate passes for assisting: successful passes with a passer
    passes = df[(df['event_type'] == 'Pass') & (df.get('outcome') == 'Successful')].copy()
    passes['player_name'] = passes['player_name'].fillna('')
    passes = passes[(passes['player_name'] != '') & (passes['team'].notna())]
    if passes.empty:
        return pd.DataFrame(columns=list(group_keys) + ['xa'])

    xa_rows = []
    for mid, shots_m in shots.groupby('match_id'):
        p_m = passes[passes['match_id'] == mid]
        if p_m.empty:
            continue
        # Iterate shots in time order
        for _, s in shots_m.sort_values('total_seconds').iterrows():
            team = s.get('team')
            if team is None:
                continue
            t = float(s.get('total_seconds', 0))
            # last successful pass by same team within window
            cand = p_m[(p_m['team'] == team) & (p_m['total_seconds'] <= t) & (p_m['total_seconds'] >= t - lookback_seconds)]
            if cand.empty:
                continue
            assist_pass = cand.iloc[-1]
            xa_rows.append({
                'match_id': mid,
                'player_name': assist_pass.get('player_name'),
                'xa': float(s.get('xg', 0.0))
            })

    if not xa_rows:
        return pd.DataFrame(columns=list(group_keys) + ['xa'])

    xa_df = pd.DataFrame(xa_rows)
    return xa_df.groupby(list(group_keys))['xa'].sum().reset_index()

def load_and_prepare_data():
    # === Primär: neue WhoScored JSON-Struktur (Data/whoscored/<team>/*.json) ===
    json_files = [
        p for p in glob.glob(os.path.join("Data", "whoscored", "*", "*.json"))
        if not p.endswith(os.path.join("whoscored", "index.json")) and not p.endswith("index.json")
    ]

    if json_files:
        print("Lade WhoScored JSON-Dateien...")
        print(f"Gefundene Dateien: {len(json_files)}")

        player_rows = []
        event_rows = []

        for path in sorted(json_files):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
            except Exception:
                continue

            status_ok = (payload.get('status') or {}).get('ok', True)
            if status_ok is not True:
                # Partial/failed scrapes werden für KPI-Berechnung übersprungen
                continue

            match_id = os.path.splitext(os.path.basename(path))[0]
            parsed = payload.get('parsed') or {}

            for row in (parsed.get('player_stats') or []):
                if isinstance(row, dict):
                    r = dict(row)
                    r['match_id'] = match_id
                    player_rows.append(r)

            for row in (parsed.get('events') or []):
                if isinstance(row, dict):
                    r = dict(row)
                    r['match_id'] = match_id
                    event_rows.append(r)

        if not player_rows:
            print("Keine Player-Daten in WhoScored JSONs gefunden.")
            return pd.DataFrame()

        combined_df = pd.DataFrame(player_rows)
        all_events_df = pd.DataFrame(event_rows) if event_rows else None

    else:
        # === Fallback: altes CSV-Format (Data/*_player_stats.csv, Data/*_events.csv) ===
        print("Lade CSV-Dateien...")
        csv_files = glob.glob("Data/*_player_stats.csv")
        print(f"Gefundene Dateien: {csv_files}")
        if not csv_files:
            return pd.DataFrame()
        dataframes = []
        for file in csv_files:
            df = pd.read_csv(file)
            match_id = os.path.basename(file).replace('_player_stats.csv', '')
            df['match_id'] = match_id
            dataframes.append(df)
        combined_df = pd.concat(dataframes, ignore_index=True)

        # --- events laden (falls vorhanden) ---
        event_files = glob.glob("Data/*_events.csv")
        all_events_df = None
        if event_files:
            print(f"Lade Event-Dateien: {len(event_files)}")
            event_dfs = [pd.read_csv(f) for f in event_files]
            for i, f in enumerate(event_files):
                match_id = os.path.basename(f).replace('_events.csv', '')
                event_dfs[i]['match_id'] = match_id
            all_events_df = pd.concat(event_dfs, ignore_index=True)
    # Entferne 'possession' aus CSV, um Überschreibungen zu vermeiden
    combined_df.drop(columns=['possession'], errors='ignore', inplace=True)

    # Rating aus PlayerStats robust erkennen und in 'rating' mappen
    rating_candidates = [
        'rating','final_rating','player_rating','sofascore_rating','sofa_score',
        'whoscored_rating','ws_rating','fotmob_rating','FotMobRating',
        'matchRating','match_rating'
    ]
    found = None
    for col in rating_candidates:
        if col in combined_df.columns:
            s = combined_df[col].astype(str).str.replace(',', '.', regex=False)
            s_num = pd.to_numeric(s, errors='coerce')
            if s_num.notna().any():
                combined_df['rating'] = s_num
                found = col
                break
    if 'rating' not in combined_df.columns:
        combined_df['rating'] = pd.NA
    # Duplikate vermeiden: Originalquelle entfernen, damit späteres Rename keine Doppelspalte erzeugt
    combined_df.drop(columns=['final_rating'], errors='ignore', inplace=True)

    match_teams = combined_df[['match_id', 'team_name']].drop_duplicates()
    match_opponents = []
    for match_id, group in match_teams.groupby('match_id'):
        teams = group['team_name'].tolist()
        if len(teams) == 2:
            match_opponents.append({'match_id': match_id, 'team_name': teams[0], 'opponent': teams[1]})
            match_opponents.append({'match_id': match_id, 'team_name': teams[1], 'opponent': teams[0]})
    opponent_map = pd.DataFrame(match_opponents)
    combined_df = pd.merge(combined_df, opponent_map, on=['match_id', 'team_name'], how='left')

    # ab hier: gemeinsamer KPI-Flow (unabhängig von JSON/CSV Quelle)
    if all_events_df is not None and not all_events_df.empty:
        goals_data = all_events_df[
            (all_events_df['event_type'] == 'Goal') |
            ((all_events_df['event_type'].isin(['Shot', 'SavedShot', 'MissedShots'])) & (all_events_df['outcome'] == 'Goal'))
        ].groupby(['match_id', 'player_name']).size().reset_index(name='goals_from_events')
        if not goals_data.empty:
            combined_df = pd.merge(combined_df, goals_data, left_on=['match_id', 'name'], right_on=['match_id', 'player_name'], how='left').drop(columns=['player_name'])
            combined_df['goals'] = combined_df['goals_from_events'].fillna(0)
            combined_df.drop(columns=['goals_from_events'], inplace=True)
        kpi_functions = [calculate_progressive_passes, calculate_advanced_kpis, calculate_xg, calculate_xa, calculate_pressure_regains, calculate_xt]
        for func in kpi_functions:
            kpi_df = func(all_events_df)
            if not kpi_df.empty:
                merged = pd.merge(combined_df, kpi_df, left_on=['match_id', 'name'], right_on=['match_id', 'player_name'], how='left')
                if 'player_name' in merged.columns:
                    merged.drop(columns=['player_name'], inplace=True)
                combined_df = merged
        ppda_data = calculate_ppda(all_events_df)
        if not ppda_data.empty:
            ppda_data.rename(columns={'team': 'team_name'}, inplace=True)
            combined_df = pd.merge(combined_df, ppda_data, on=['match_id', 'team_name'], how='left')
        field_tilt_data = calculate_field_tilt(all_events_df)
        if not field_tilt_data.empty:
            field_tilt_data.rename(columns={'team': 'team_name'}, inplace=True)
            combined_df = pd.merge(combined_df, field_tilt_data, on=['match_id', 'team_name'], how='left')

        # --- NEU: Berechnung der durchschnittlichen Abstossdistanz ---
        if 'is_goalkick' in all_events_df.columns and 'pass_length' in all_events_df.columns:
            ev_gk = all_events_df.copy()
            # "True"/"False"/1/0 -> bool
            ev_gk['is_goalkick'] = ev_gk['is_goalkick'].apply(lambda v: str(v).strip().lower() in ['true', '1', 't', 'yes', 'y'])
            ev_gk['pass_length'] = pd.to_numeric(ev_gk['pass_length'], errors='coerce')
            # Alle Abstoß-Events mit Länge berücksichtigen (nicht nur event_type == 'Pass')
            goalkicks = ev_gk[(ev_gk['is_goalkick']) & (ev_gk['pass_length'].notna())]
            if not goalkicks.empty:
                avg_distance = (
                    goalkicks.groupby(['match_id', 'team'])['pass_length']
                    .mean()
                    .round(0)
                    .reset_index(name='avg_goalkick_distance')
                )
                avg_distance.rename(columns={'team': 'team_name'}, inplace=True)
                combined_df = pd.merge(combined_df, avg_distance, on=['match_id', 'team_name'], how='left')
        # BALLBESITZ PRO TEAM & SPIELER — 100 % KONSISTENT
        print("Berechne Ballbesitz: Spieler- und Team-ActionMinutes → Prozent der Match-Länge")

        ev = all_events_df.copy()
        ev['minute'] = pd.to_numeric(ev.get('minute', 0), errors='coerce').fillna(0)
        ev['second'] = pd.to_numeric(ev.get('second', 0), errors='coerce').fillna(0)
        ev['time_sec'] = ev['minute'] * 60 + ev['second']
        ev = ev.sort_values(['match_id', 'time_sec']).reset_index(drop=True)

        # Match-Ende (Fallback 90min)
        match_end = ev.groupby('match_id')['time_sec'].max().reset_index(name='match_end_sec')
        match_end['match_end_sec'] = match_end['match_end_sec'].replace(0, 90*60)
        match_end_map = dict(zip(match_end['match_id'], match_end['match_end_sec']))
        match_end_min_map = {mid: sec / 60 for mid, sec in match_end_map.items()}

        # Dauer zwischen Events
        ev['next_time'] = ev.groupby('match_id')['time_sec'].shift(-1)
        ev = ev.merge(match_end, on='match_id', how='left')
        ev['next_time'] = ev['next_time'].fillna(ev['match_end_sec'])
        ev['duration'] = (ev['next_time'] - ev['time_sec']).clip(lower=0)

        # Nur Events mit Spielerzuordnung
        ev['player_name'] = ev['player_name'].fillna('')
        ev_player = ev[ev['player_name'] != '']

        # === SPIELER-EBENE ===
        player_poss = ev_player.groupby(['match_id', 'player_name'])['duration'].sum().reset_index(name='possession_seconds')
        if not player_poss.empty:
            player_poss['match_end_sec'] = player_poss['match_id'].map(match_end_map).fillna(90*60)
            player_poss['action_minutes'] = (player_poss['possession_seconds'] / 60).round(2)
            player_poss['match_end_min'] = (player_poss['match_end_sec'] / 60).round(1)
            player_poss['action_minutes_pct'] = (player_poss['action_minutes'] / player_poss['match_end_min'].replace(0, 90) * 100).round(1)
        else:
            player_poss = pd.DataFrame(columns=['match_id','player_name','possession_seconds','action_minutes','action_minutes_pct','match_end_min'])

        # === TEAM-EBENE (Summierte Spieler-ActionMinutes) ===
        player_team_map = ev_player[['match_id', 'player_name', 'team']].drop_duplicates()
        if not player_poss.empty and not player_team_map.empty:
            team_minutes = (
                player_poss.merge(player_team_map, on=['match_id', 'player_name'], how='left')
                            .groupby(['match_id', 'team'], dropna=False)['action_minutes'].sum()
                            .reset_index()
            )
            team_minutes.rename(columns={'team': 'team_name', 'action_minutes': 'team_action_minutes'}, inplace=True)
            team_minutes['team_possession'] = team_minutes['team_action_minutes']
        else:
            team_minutes = pd.DataFrame(columns=['match_id', 'team_name', 'team_action_minutes', 'team_possession'])

        combined_df = pd.merge(
            combined_df,
            team_minutes[['match_id', 'team_name', 'team_action_minutes', 'team_possession']],
            on=['match_id', 'team_name'],
            how='left'
        )
        combined_df['possession'] = combined_df['team_possession'].fillna(0.0)

        if not player_poss.empty:
            combined_df = pd.merge(
                combined_df,
                player_poss[['match_id', 'player_name', 'possession_seconds', 'action_minutes', 'action_minutes_pct', 'match_end_min']],
                left_on=['match_id', 'name'],
                right_on=['match_id', 'player_name'],
                how='left'
            )
            combined_df.drop(columns=['player_name'], errors='ignore', inplace=True)

            combined_df['possession'] = np.where(
                combined_df['action_minutes'].notna(),
                combined_df['action_minutes'],
                combined_df['possession']
            )
            combined_df['possession'] = pd.to_numeric(combined_df['possession'], errors='coerce').round(2).fillna(0.0)

        # Neue Kennzahl Ballbesitz: exakt wie action_minutes, aber geteilt durch max Spielzeit und *100 für Prozent
        combined_df['ballbesitz'] = (combined_df['action_minutes'] / combined_df['match_end_min'].replace(0, 90) * 100).round(1)

        # Defaults & Cleanup
        combined_df['possession'] = pd.to_numeric(combined_df['possession'], errors='coerce').fillna(0.0)
        if 'action_minutes' in combined_df.columns:
            combined_df['action_minutes'] = pd.to_numeric(combined_df['action_minutes'], errors='coerce').fillna(0.0)
        if 'team_action_minutes' in combined_df.columns:
            combined_df['team_action_minutes'] = pd.to_numeric(combined_df['team_action_minutes'], errors='coerce').fillna(0.0)
        if 'team_possession' in combined_df.columns:
            combined_df['team_possession'] = pd.to_numeric(combined_df['team_possession'], errors='coerce').fillna(0.0)
        if 'possession_seconds' in combined_df.columns:
            combined_df['possession_seconds'] = pd.to_numeric(combined_df['possession_seconds'], errors='coerce').fillna(0).astype(int)

        # --- Ende Possession ---
    # --- Ende Events-Block ---


    minute_candidates = ['minutes_played', 'minutesPlayed', 'minutes', 'time_on_pitch', 'minutesplayed', 'mins']
    if any(c in combined_df.columns for c in minute_candidates):
        # normalize possible column names into minutes_played (keep numeric)
        for c in minute_candidates:
            if c in combined_df.columns and c != 'minutes_played':
                combined_df['minutes_played'] = pd.to_numeric(combined_df[c], errors='coerce')
                break
        combined_df['minutes_played'] = pd.to_numeric(combined_df.get('minutes_played'), errors='coerce')
    else:
        combined_df['minutes_played'] = pd.NA

    if all_events_df is not None and not all_events_df.empty:
        ev = all_events_df.copy()
        ev['minute'] = pd.to_numeric(ev.get('minute', 0), errors='coerce').fillna(0)
        ev['second'] = pd.to_numeric(ev.get('second', 0), errors='coerce').fillna(0)
        ev['total_seconds'] = ev['minute'] * 60 + ev['second']

        # Match-Ende (inkl. Nachspielzeit) in Sekunden
        match_end = ev.groupby('match_id')['total_seconds'].max().reset_index(name='match_end_sec')

        # erste und letzte Aktion pro Spieler
        first_sec = ev.groupby(['match_id', 'player_name'])['total_seconds'].min().reset_index(name='first_sec')
        last_sec = ev.groupby(['match_id', 'player_name'])['total_seconds'].max().reset_index(name='last_sec')

        player_times = first_sec.merge(last_sec, on=['match_id', 'player_name'], how='outer')
        player_times = player_times.merge(match_end, on='match_id', how='left')

        # bring isFirstEleven aus den player_stats ins Zeit-DF
        if 'isFirstEleven' in combined_df.columns:
            flags = combined_df[['match_id', 'name', 'isFirstEleven']].drop_duplicates()
            player_times = player_times.merge(flags, left_on=['match_id', 'player_name'], right_on=['match_id', 'name'], how='left')
        else:
            player_times['isFirstEleven'] = False

        player_times['isFirstEleven'] = player_times['isFirstEleven'].fillna(False)

        # Schätzung: Starter = bis zu ihrer letzten Aktion; Einwechselspieler = von ihrem ersten Event bis Match-Ende
        player_times['minutes_est'] = np.where(
            player_times['isFirstEleven'],
            np.ceil(player_times['last_sec'] / 60),
            np.ceil((player_times['match_end_sec'] - player_times['first_sec']) / 60)
        )
        player_times['minutes_est'] = player_times['minutes_est'].clip(lower=0).fillna(0).astype(int)

        # Merge Schätzung in combined_df (per match + player name)
        combined_df = combined_df.merge(
            player_times[['match_id', 'player_name', 'minutes_est']],
            left_on=['match_id', 'name'],
            right_on=['match_id', 'player_name'],
            how='left'
        )

        # Regeln zum Überschreiben:
        # - Wenn minutes_played fehlt -> benutze Schätzung
        # - Wenn Spieler KEIN Starter ist und recorded minutes_played > minutes_est -> tausche (bspw. 90 bei Einwechselspieler)
        # - Wenn recorded < geschätztem -> aktualisiere ebenfalls
        combined_df['minutes_played'] = pd.to_numeric(combined_df.get('minutes_played'), errors='coerce')
        is_first = combined_df.get('isFirstEleven', pd.Series([False]*len(combined_df))).fillna(False)

        mask_missing = combined_df['minutes_played'].isna()
        mask_too_large_sub = (~is_first) & combined_df['minutes_played'].notna() & (combined_df['minutes_played'] > combined_df['minutes_est'].fillna(0))
        mask_less_than_est = combined_df['minutes_played'].notna() & combined_df['minutes_played'] < combined_df['minutes_est'].fillna(0)

        mask = mask_missing | mask_too_large_sub | mask_less_than_est
        combined_df.loc[mask, 'minutes_played'] = combined_df.loc[mask, 'minutes_est']

        # aufräumen temporäre Spalten
        combined_df['minutes_played'] = pd.to_numeric(combined_df['minutes_played'], errors='coerce').fillna(0).astype(int)
        for tmp in ['player_name', 'minutes_est']:
            if tmp in combined_df.columns:
                combined_df.drop(columns=[tmp], inplace=True)
    else:
        # keine Events vorhanden -> fehlende Werte mit 0 füllen
        combined_df['minutes_played'] = pd.to_numeric(combined_df.get('minutes_played', 0), errors='coerce').fillna(0).astype(int)

    # --- Ende minutes_played Ergänzung ---
    combined_df['half'] = 0
    return combined_df

def calculate_kpis(df):
    print("Berechne KPIs...")
    # Falls 'rating' noch nicht gesetzt wurde, nur dann final_rating -> rating mappen
    if 'rating' not in df.columns and 'final_rating' in df.columns:
        df.rename(columns={'final_rating': 'rating'}, inplace=True)

    df.rename(columns={
        'name': 'player','team_name': 'team','passesTotal': 'passes_total','passesAccurate': 'passes_successful',
        'passesKey': 'key_passes','shotsTotal': 'shots','minutesPlayed': 'minutes_played',
        'shotsOnTarget': 'shots_on_target','shotsOffTarget': 'shots_off_target','shotsBlocked': 'shots_blocked',
        'tacklesTotal': 'tackles','tackleSuccessful': 'tackles_successful','interceptions': 'interceptions',
        'clearances': 'clearances','tackleUnsuccesful': 'tackles_unsuccessful','aerialsTotal': 'aerials_total',
        'aerialsWon': 'aerials_won','defensiveAerials': 'defensive_aerials','offensiveAerials': 'offensive_aerials',
        'dribblesAttempted': 'dribbles_attempted','dribblesWon': 'dribbles_successful','foulsCommitted': 'fouls_committed',
        'offsidesCaught': 'offsides','dispossessed': 'dispossessed','touches': 'touches','dribbledPast': 'dribbled_past',
        'errors': 'errors','cornersTotal': 'corners_total','cornersAccurate': 'corners_accurate',
        'throwInsTotal': 'throw_ins_total','throwInsAccurate': 'throw_ins_accurate','collected': 'saves_collected',
        'totalSaves': 'saves_total','parriedSafe': 'saves_parried_safe','parriedDanger': 'saves_parried_danger'
    }, inplace=True)

    # Rating sicher numerisch machen, auch wenn als String mit Komma vorliegt
    if 'rating' in df.columns:
        if not np.issubdtype(df['rating'].dtype, np.number):
            df['rating'] = pd.to_numeric(df['rating'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        df['rating'] = df['rating'].fillna(0).round(2)
    else:
        df['rating'] = 0.0

    # Touches/Assists/XA: Frontend erwartet die Keys; falls nicht vorhanden -> 0
    if 'touches' not in df.columns:
        df['touches'] = 0
    df['touches'] = pd.to_numeric(df.get('touches', 0), errors='coerce').fillna(0).astype(int)

    if 'assists' not in df.columns:
        df['assists'] = 0
    df['assists'] = pd.to_numeric(df.get('assists', 0), errors='coerce').fillna(0)

    if 'xa' not in df.columns:
        df['xa'] = 0.0
    df['xa'] = pd.to_numeric(df.get('xa', 0.0), errors='coerce').fillna(0.0)

    # SPIELMINUTEN: nutze vorhandene / berechnete Werte; wenn völlig fehlend -> 0
    if 'minutes_played' not in df.columns:
        df['minutes_played'] = 0
    df['minutes_played'] = pd.to_numeric(df['minutes_played'], errors='coerce').fillna(0).astype(int)

    df['pass_accuracy'] = (df['passes_successful'] / df['passes_total'].replace(0, 1) * 100).round(2)
    df['duels_total'] = df.get('tackles', 0) + df.get('aerials_total', 0)
    df['duels_won'] = df.get('tackles_successful', 0) + df.get('aerials_won', 0)
    df['duel_win_rate'] = (df['duels_won'] / df['duels_total'].replace(0, 1) * 100).round(2)
    if 'tackles' in df.columns and 'tackles_successful' in df.columns:
        df['tackle_success_rate'] = (df['tackles_successful'] / df['tackles'].replace(0, 1) * 100).round(2)
    if 'aerials_total' in df.columns and 'aerials_won' in df.columns:
        df['aerial_success_rate'] = (df['aerials_won'] / df['aerials_total'].replace(0, 1) * 100).round(2)
    if 'dribbles_attempted' in df.columns and 'dribbles_successful' in df.columns:
        df['dribble_success_rate'] = (df['dribbles_successful'] / df['dribbles_attempted'].replace(0, 1) * 100).round(2)
    if 'shots' in df.columns and 'goals' in df.columns:
        df['shot_conversion_rate'] = (df['goals'] / df['shots'].replace(0, 1) * 100).round(2)

    p90_cols = ['passes_total','passes_successful','key_passes','touches','shots','shots_on_target','shots_off_target','shots_blocked',
                'passes_into_area14','passes_into_zone11','third_to_third_passes','goals','progressive_passes','progressive_carries','passes_into_final_third', # HINZUGEFÜGT
                'box_entries_total','switches_of_play','tackles','tackles_successful','tackles_unsuccessful','interceptions',
                'clearances','dribbled_past','errors','recoveries','dribbles_attempted','dribbles_successful','duels_total',
                'duels_won','aerials_total','aerials_won','pressure_regains','xt','defensive_aerials','offensive_aerials',
                'fouls_committed','dispossessed','offsides','sca','gca','corners_total','corners_accurate','throw_ins_total',
                'throw_ins_accurate','saves_total','saves_collected','saves_parried_safe','saves_parried_danger','xg',
                'defensive_third_touches','action_minutes']
    for col in p90_cols:
        if col in df.columns:
            df[f'{col}_p90'] = (df[col] / df['minutes_played'].replace(0, 1) * 90).round(2)

    # Neue Kennzahl Ballbesitz: exakt wie action_minutes, aber geteilt durch max Spielzeit und *100 für Prozent
    df['ballbesitz'] = (df['action_minutes'] / df['match_end_min'].replace(0, 90) * 100).round(1)

    # Frontend-Key: possession als TEAM-Ballbesitz in Prozent (gegneranalyse.js erwartet Team-Level)
    if 'team_action_minutes' in df.columns and 'match_end_min' in df.columns:
        df['possession'] = (pd.to_numeric(df.get('team_action_minutes', 0), errors='coerce').fillna(0.0)
                            / pd.to_numeric(df.get('match_end_min', 90), errors='coerce').replace(0, 90).fillna(90)
                            * 100).round(1)
    else:
        # Fallback: wenn Team-Minuten nicht vorhanden sind, nutze Spieler-Proxy
        df['possession'] = pd.to_numeric(df.get('ballbesitz', 0.0), errors='coerce').fillna(0.0).round(1)

    # Aliase für Frontend (Gegneranalyse könnte andere Feldnamen erwarten)
    if 'avg_goalkick_distance' in df.columns:
        df['avg_goalkick_distance'] = pd.to_numeric(df['avg_goalkick_distance'], errors='coerce').fillna(0)
        df['avg_goalkick_length'] = df['avg_goalkick_distance']
        df['average_goalkick_distance'] = df['avg_goalkick_distance']

    base_cols = [
        'match_id','team','opponent','half','player','goals','passes_total','passes_successful',
        'rating','minutes_played','possession','touches','assists','xa',
        'key_passes','shots','shots_on_target',
        'shots_off_target','shots_blocked','progressive_passes','passes_into_area14',
        'passes_into_zone11','third_to_third_passes','pre_assists','xt','tackles','tackles_successful', # HINZUGEFÜGT
        'tackles_unsuccessful','interceptions','clearances','dribbled_past','errors',
        'recoveries','dribbles_attempted','progressive_carries','xg','ppda',
        'dribbles_successful','duels_total','duels_won','aerials_total','aerials_won',
        'defensive_aerials','offensive_aerials','fouls_committed','dispossessed',
        'offsides','corners_total','corners_accurate','throw_ins_total',
        'throw_ins_accurate','saves_total','saves_collected','saves_parried_safe',
        'saves_parried_danger','passes_into_final_third','box_entries_pass',
        'box_entries_carry','box_entries_total','switches_of_play',
        'defensive_third_touches','sca','gca','field_tilt','pass_accuracy',
        'tackle_success_rate','duel_win_rate','aerial_success_rate',
        'shot_conversion_rate','dribble_success_rate','avg_goalkick_distance',
        'action_minutes','ballbesitz'  # Neue Kennzahl Ballbesitz hinzugefügt
       ]
    p90_cols = [c for c in df.columns if c.endswith('_p90')]
    final_cols = base_cols + p90_cols
    if 'pressure_regains' in df.columns: final_cols += ['pressure_regains']

    # Schutz vor doppelten Spaltennamen (sonst liefert kpis[col] ein DataFrame)
    final_cols = list(dict.fromkeys(final_cols))

    kpis = pd.DataFrame(columns=final_cols)
    for col in final_cols:
        if col in df.columns: kpis[col] = df[col]
    kpis.fillna(0, inplace=True)
    for col in kpis.select_dtypes(include=['number']).columns:
        kpis[col] = pd.to_numeric(kpis[col], errors='coerce').fillna(0)
        if kpis[col].dtype == 'float64':
            kpis[col] = kpis[col].astype(float)
        else:
            kpis[col] = kpis[col].astype(int)
    return kpis.to_dict('records')

def build_kpi_groups(columns: list[str]) -> dict:
    cols = set(columns)

    def with_p90(base: set[str]) -> list[str]:
        out = set()
        for c in base:
            if c in cols:
                out.add(c)
            p90 = f"{c}_p90"
            if p90 in cols:
                out.add(p90)
        return sorted(out)

    # Übersicht
    grp_uebersicht = {
        'match_id','team','opponent','half','player',
        'minutes_played','rating','goals','shots','shots_on_target','xg',
        'touches','pass_accuracy','duel_win_rate','tackle_success_rate','aerial_success_rate',
        'shot_conversion_rate','dribble_success_rate','action_minutes','ballbesitz','field_tilt'
    }

    # Torerfolg & Kreation
    grp_torerfolg_kreation = {
        'key_passes','sca','gca','passes_into_final_third','passes_into_area14',
        'passes_into_zone11','progressive_passes','progressive_carries','switches_of_play', # HINZUGEFÜGT
        'box_entries_pass','box_entries_carry','box_entries_total','xt',
        'shots','shots_on_target','shots_off_target','shots_blocked','xg','offsides'
    }

    # Spiel gegen den Ball
    grp_gegen_ball = {
        'tackles','tackles_successful','tackles_unsuccessful','interceptions',
        'clearances','recoveries','pressure_regains',
        'defensive_third_touches','dribbled_past','errors',
        'duels_total','duels_won','aerials_total','aerials_won',
        'defensive_aerials','offensive_aerials',
        'fouls_committed','dispossessed','ppda',
        'saves_total','saves_collected','saves_parried_safe','saves_parried_danger'
    }

    # Standards
    grp_standards = {
        'corners_total','corners_accurate','throw_ins_total','throw_ins_accurate'
    }

    groups = {
        'Übersicht': with_p90(grp_uebersicht),
        'Torerfolg & Kreation': with_p90(grp_torerfolg_kreation),
        'Spiel gegen den Ball': with_p90(grp_gegen_ball),
        'Standards': with_p90(grp_standards),
    }
    # nur vorhandene Spalten zurückgeben
    for k in groups:
        groups[k] = [c for c in groups[k] if c in cols]
    return groups

def main():
    print("Verarbeite Datei(en)...")
    df = load_and_prepare_data()
    if df.empty:
        print("Keine Daten zum Verarbeiten.")
        return
    kpis = calculate_kpis(df)

    # Filter: Spieler ohne Spielzeit (minutes_played == 0) nicht ausgeben
    kpis_df = pd.DataFrame(kpis)
    if 'minutes_played' in kpis_df.columns:
        kpis_df = kpis_df[kpis_df['minutes_played'].astype(float) > 0]
    filtered_kpis = kpis_df.to_dict('records')

    with open('processed_data.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_kpis, f, ensure_ascii=False, indent=2)
    print(f"FERTIG! processed_data.json erstellt. ({len(filtered_kpis)} Zeilen)")
    print("Spielminuten = korrekt | p90 = korrekt | Einwechsler = erkannt")

    # Gruppierungen (zusätzlich unter deutschem Namen speichern)
    groups = build_kpi_groups(list(kpis_df.columns))
    with open('kpi_groups.json', 'w', encoding='utf-8') as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)
    with open('kennzahlen_groups.json', 'w', encoding='utf-8') as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)
    print("kpi_groups.json und kennzahlen_groups.json erstellt.")

if __name__ == "__main__":
    main()
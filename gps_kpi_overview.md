# GPS / StatsSports – benötigte Kennzahlen (pro Trainingsinhalt / Cut)

Diese Liste ist auf Basis der StatsSports-CSV-Vorlage in `Data/GPS/20240316_27c44cee-5332-4111-ad14-42965dee59c8.csv` erstellt (1660 Spalten). Unten ist jeweils das **exakte Spalten-Mapping** angegeben.

## 1) Basis-KPIs (direkt vorhanden)
- Datum: `date`
- Dauer: `total_duration` (Sekunden)
- Total Distance: `total_distance` (Meter)
- Meter pro Minute: `meterage_per_minute` (m/min)
- Total PlayerLoad: `total_player_load`
- Peak PlayerLoad: `peak_player_load`
- PlayerLoad pro Minute: `player_load_per_minute`
- Max Speed: `max_vel`

## 2) Distance in Zone 1–5 (Speed-Zonen)
Direkt vorhanden als Distanz pro Geschwindigkeitszone:
- Zone 1: `velocity_band1_total_distance`
- Zone 2: `velocity_band2_total_distance`
- Zone 3: `velocity_band3_total_distance`
- Zone 4: `velocity_band4_total_distance`
- Zone 5: `velocity_band5_total_distance`

Hinweis: Es existieren zusätzlich Zone 6–8 (`velocity_band6_total_distance` …), falls du später erweitern willst.

## 3) Beschleunigungen – Anzahl Zone 1–5
Direkt vorhanden als Effort-Count pro Accel-Zone:
- Zone 1: `acceleration_band1_total_effort_count`
- Zone 2: `acceleration_band2_total_effort_count`
- Zone 3: `acceleration_band3_total_effort_count`
- Zone 4: `acceleration_band4_total_effort_count`
- Zone 5: `acceleration_band5_total_effort_count`

## 4) Beschleunigungen – Distanz Zone 1–5
Direkt vorhanden als Distanz pro Accel-Zone:
- Zone 1: `acceleration_band1_total_distance`
- Zone 2: `acceleration_band2_total_distance`
- Zone 3: `acceleration_band3_total_distance`
- Zone 4: `acceleration_band4_total_distance`
- Zone 5: `acceleration_band5_total_distance`

## 5) Entschleunigungen – Anzahl Zone 1–5
Wichtig: In der vorliegenden Vorlage gibt es **keine** `deceleration_band1_total_*` … `deceleration_band5_total_*` Felder.

Was es aber gibt (und was zeigt, dass „Entschleunigung“ in StatsSports i.d.R. als **negative** Werte auftaucht):
- `max_effort_deceleration` ist **negativ** (z.B. median ca. -5.08 in der Beispieldatei)
- `deceleration_per_minute` (Gesamt-Entschleunigungen pro Minute, nicht in Zonen 1–5)

Vorhandene Alternativen (nicht zoniert in 1–5):
- `deceleration_per_minute`
- `max_effort_deceleration`

Teilweise zonierte/andere Systematik (IMA, nur Band 1–3):
- `ima_band1_decel_count`, `ima_band2_decel_count`, `ima_band3_decel_count`

Wenn du zwingend „Decel Zone 1–5“ brauchst, müssen wir das entweder:
1) als **eigene abgeleitete KPIs** definieren (und später beim FakeGPS befüllen), oder
2) die Decel-Zonen in StatsSports anders mappen (wenn du mir sagst, welche Spalten in euren echten Exports dafür verwendet werden).

## 6) Entschleunigungen – Distanz Zone 1–5
Gleiche Situation wie bei den Counts: keine `deceleration_band*` Distanzspalten in der Vorlage.

## 7) Metabolic Distanz Zone 1–5
Direkt vorhanden als Metabolic-Power-Bands (Distanz):
- Zone 1: `metabolic_power_band1_total_distance`
- Zone 2: `metabolic_power_band2_total_distance`
- Zone 3: `metabolic_power_band3_total_distance`
- Zone 4: `metabolic_power_band4_total_distance`
- Zone 5: `metabolic_power_band5_total_distance`

Hinweis: Es existieren auch Zone 6–8.

## Nächster Schritt
Ich habe als nächstes eine KPI-Spezifikation als JSON angelegt, damit wir später pro Trainingsinhalt/Cut exakt wissen: Key, Einheit, Quellspalte, Aggregation.
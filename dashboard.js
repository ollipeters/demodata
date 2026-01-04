document.addEventListener('DOMContentLoaded', () => {
    // --- DOM ---
    const kpiDropdown = document.getElementById('kpi-dropdown-container');
    const kpiButton = document.getElementById('kpi-dropdown-button');
    const kpiCheckboxes = document.getElementById('kpi-checkboxes');
    const kpiText = document.getElementById('kpi-selection-text');
    const clearKpi = document.getElementById('clear-kpi');

    const groupbyDropdown = document.getElementById('groupby-dropdown-container');
    const groupbyButton = document.getElementById('groupby-dropdown-button');
    const groupbyCheckboxes = document.getElementById('groupby-checkboxes');
    const groupbyText = document.getElementById('groupby-selection-text');
    const clearGroupby = document.getElementById('clear-groupby');

    const tableTitle = document.getElementById('table-title');
    const tableHeader = document.getElementById('table-header');
    const tableBody = document.getElementById('table-body');

    // --- State ---
    let ALL_KPIS = [];
    let dataset = [];
    // Standardauswahl beim Laden der Seite
let selectedKpis = [
        { value: 'goals', label: 'goals' },
        { value: 'xg', label: 'xG' },
        { value: 'passes_total', label: 'passes' },
        { value: 'pass_accuracy', label: 'pass_accuracy' },
        { value: 'passes_into_final_third', label: 'passes_into_final_third' },
        { value: 'passes_into_zone14', label: 'passes_into_zone14' },
        { value: 'shots_on_target', label: 'shots_on_target' },
        { value: 'tackle_success_rate', label: 'tackle_success_rate' },
        { value: 'ppda', label: 'ppda' },
        { value: 'field_tilt', label: 'field_tilt' }
    ];
    let selectedGroups = [
        { value: 'match_id', label: 'Match' },
        { value: 'team', label: 'Team' },
        { value: 'player', label: 'Spieler' }
    ];
    let expandedRows = new Set(); // Speichert die IDs der aufgeklappten Zeilen
    let filters = {
        team: new Set(),
        opponent: new Set(),
        player: new Set(),
        match_id: new Set()
    };

    const GROUPING_OPTIONS = [
        { value: 'match_id', label: 'Match' },
        { value: 'team', label: 'Team' },
        { value: 'opponent', label: 'Gegner' },
        { value: 'player', label: 'Spieler' }
    ];

const KPI_CATEGORIES = {
    // 1. ALLGEMEIN – Sofort-Infos für Trainer
    "Allgemein": [
        "rating", "minutes_played", "possession", "touches",
        "goals", "assists", "xg", "xa", "shot_conversion_rate",
        "action_minutes", "ballbesitz"
    ],

    // 2. SPIEL MIT DEM BALL – Aufbau, Tempo, Kreativität
    "Spiel mit dem Ball": [
        // Pässe
        "passes_total", "passes_successful", "pass_accuracy",
        "key_passes", "progressive_passes", "xt",
        "passes_into_final_third", "passes_into_area14",
        "third_to_third_passes", "switches_of_play","passes_into_zone11",

        // Abschluss & Kreation
        "shots", "shots_on_target", "shots_off_target", "shots_blocked",
        "sca", "gca", "pre_assists",

        // Dribbling & Carries
        "dribbles_attempted", "dribbles_successful", "dribble_success_rate",
        "progressive_carries", "box_entries_carry", "box_entries_pass", "box_entries_total"
    ],

    // 3. SPIEL GEGEN DEN BALL – Pressing, Zweikämpfe, Verteidigung
    "Spiel gg den Ball": [
        // Zweikämpfe
        "duels_total", "duels_won", "duel_win_rate",
        "tackles", "tackles_successful", "tackles_unsuccessful", "tackle_success_rate",
        "aerials_total", "aerials_won", "aerial_success_rate",
        "defensive_aerials", "offensive_aerials",

        // Defensivaktionen
        "interceptions", "clearances", "recoveries",
        "pressure_regains", "dribbled_past",
        "defensive_third_touches",

        // Team-Pressing
        "ppda", "field_tilt"
    ],

    // 4. STANDARDS – Ecken, Einwürfe, Freistöße
    "Standards": [
        "corners_total", "corners_accurate",
        "throw_ins_total", "throw_ins_accurate",
        "fouls_drawn", "offsides"
    ],

    // 5. TORWART & DISZIPLIN – Keeper + Fehler
    "Torwart & Disziplin": [
        "saves_total", "saves_collected", "saves_parried_safe", "saves_parried_danger",
        "fouls_committed", "dispossessed", "errors"
    ],

    // 6. STATS /90 – Faire Vergleiche (Einwechsler inkl.)
    "Stats /90": [
        "touches_p90", "action_minutes_p90",
        "goals_p90", "xg_p90", "assists_p90", "pre_assists_p90",
        "sca_p90", "gca_p90",
        "shots_p90", "shots_on_target_p90", "shots_off_target_p90", "shots_blocked_p90",
        "passes_total_p90", "passes_successful_p90", "key_passes_p90",
        "progressive_passes_p90", "xt_p90",
        "passes_into_final_third_p90", "passes_into_area14_p90",
        "third_to_third_passes_p90", "switches_of_play_p90",
        "box_entries_total_p90",
        "dribbles_attempted_p90", "dribbles_successful_p90",
        "duels_total_p90", "duels_won_p90",
        "tackles_p90", "tackles_successful_p90", "tackles_unsuccessful_p90",
        "aerials_total_p90", "aerials_won_p90",
        "defensive_aerials_p90", "offensive_aerials_p90",
        "interceptions_p90", "clearances_p90", "pressure_regains_p90",
        "dribbled_past_p90", "defensive_third_touches_p90",
        "saves_total_p90", "saves_collected_p90",
        "saves_parried_safe_p90", "saves_parried_danger_p90",
        "fouls_committed_p90", "dispossessed_p90", "offsides_p90", "errors_p90",
        "corners_total_p90", "corners_accurate_p90",
        "throw_ins_total_p90", "throw_ins_accurate_p90","passes_into_zone11_p90"
    ]
};

    // --- UI Helpers ---
    function createKpiCategoryHTML(categoryName, kpiKeys, allKpis) {
        const itemsHtml = kpiKeys
            .map(key => allKpis.find(k => k.value === key))
            .filter(Boolean) // Filtere KPIs, die evtl. nicht in den Daten sind
            .map(kpi => `
                <label class="flex items-center text-xs pl-5 py-1">
                    <input type="checkbox" class="kpi-cb mr-2" value="${kpi.value}" data-label="${kpi.label}">
                    <span>${kpi.label}</span>
                </label>`).join('');

        return `
            <div class="kpi-category">
                <div class="font-semibold text-xs cursor-pointer p-1 hover:bg-gray-100 rounded-md flex items-center">
                    <svg class="chevron" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" /></svg>
                    <span>${categoryName}</span>
                </div>
                <div class="kpi-items hidden pl-2">${itemsHtml}</div>
            </div>`;
    }

    // --- Helper ---
    function formatKpiLabel(key) {
        return key
            .replace(/_/g, ' ')
            .replace(/\b(?!p90)\w/g, l => l.toUpperCase()) // p90 klein lassen
            .replace('Passquote', 'Passquote (%)')
            .replace('Xg', 'xG')
            .replace('Xa', 'xA');
    }

    function formatMatchId(matchId, data = dataset) {
        // Finde den ersten Eintrag für dieses Match im Datensatz
        const matchInfo = data.find(row => row.match_id === matchId);
        if (matchInfo) {
            // Bestimme, wer Heim- und Auswärtsteam ist
            const homeTeam = matchInfo.team.includes('(Home)') ? matchInfo.team.replace(' (Home)', '') : matchInfo.opponent;
            const awayTeam = matchInfo.team.includes('(Home)') ? matchInfo.opponent : matchInfo.team.replace(' (Away)', '');
            return `${homeTeam} - ${awayTeam}`;
        }
        // Fallback, falls kein Match gefunden wird
        return matchId.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    // --- Load Data ---
    async function loadData() {
        try {
            const res = await fetch('processed_data.json');
            if (!res.ok) throw new Error('processed_data.json nicht gefunden');
            dataset = await res.json();

            // N/A-Zeilen komplett entfernen
            dataset = dataset.filter(row =>
                row.player !== 'N/A' &&
                row.opponent !== 'N/A' &&
                row.player !== null &&
                row.opponent !== null
            );

            const numericKeys = Object.keys(dataset[0]).filter(k =>
                typeof dataset[0][k] === 'number' && !['half'].includes(k)
            );
            // Filtere die spezielle Kennzahl hier heraus
            ALL_KPIS = numericKeys.filter(k => k !== 'avg_goalkick_distance').map(k => ({ value: k, label: formatKpiLabel(k) }));
            // Korrigiere das Label für 'goals' manuell, falls es anders formatiert wurde
            const goalsKpi = ALL_KPIS.find(k => k.value === 'goals');
            if(goalsKpi) goalsKpi.label = 'Tore';


            initialize();
        } catch (err) {
            tableBody.innerHTML = `<tr><td class="p-4 text-center text-red-500">Fehler: ${err.message}</td></tr>`;
        }
    }

    // --- Init ---
    function initialize() {
        createKpiCheckboxes();
        createGroupbyCheckboxes();
        setupFilters();
        addEventListeners();

        // Setze die Checkboxen für die Standardauswahl
        selectedKpis.forEach(kpi => {
            const cb = kpiCheckboxes.querySelector(`input[value="${kpi.value}"]`);
            if (cb) cb.checked = true;
        });
        selectedGroups.forEach(group => {
            const cb = groupbyCheckboxes.querySelector(`input[value="${group.value}"]`);
            if (cb) cb.checked = true;
        });
        updateTable();
    }

    function createKpiCheckboxes() {
        const categorizedHtml = Object.entries(KPI_CATEGORIES)
            .map(([category, keys]) => createKpiCategoryHTML(category, keys, ALL_KPIS))
            .join('');

        // Finde alle KPIs, die in keiner Kategorie sind
        const categorizedKpis = new Set(Object.values(KPI_CATEGORIES).flat());
        const uncategorizedHtml = ALL_KPIS
            .filter(kpi => !categorizedKpis.has(kpi.value))
            .map(kpi => `
                <label class="flex items-center text-xs p-1">
                    <input type="checkbox" class="kpi-cb mr-2" value="${kpi.value}" data-label="${kpi.label}">
                    <span>${kpi.label}</span>
                </label>
            `).join('');

        kpiCheckboxes.innerHTML = categorizedHtml + uncategorizedHtml;
    }

    function createGroupbyCheckboxes() {
        groupbyCheckboxes.innerHTML = GROUPING_OPTIONS.map(g => `
            <label class="flex items-center text-xs">
                <input type="checkbox" class="groupby-cb mr-2" value="${g.value}" data-label="${g.label}">
                <span>${g.label}</span>
            </label>
        `).join('');
    }

    function setupFilters() {
        ['team', 'opponent', 'player', 'match_id'].forEach(key => {
            const container = document.getElementById(`${key}-checkboxes`);
            const search = container.querySelector('.search-input');

            const values = [...new Set(dataset.map(r => r[key]).filter(v => v && v !== 'N/A'))].sort();
            // Füge die Labels direkt in den Container ein, nicht in ein extra div
            const itemsHTML = values.map(v => `
                <label class="flex w-full items-center text-xs py-1">
                    <input type="checkbox" class="filter-cb mr-2" data-key="${key}" value="${v}" data-display-name="${key === 'match_id' ? formatMatchId(v) : v}">
                    <span>${key === 'match_id' ? formatMatchId(v) : v}</span>
                </label>
            `).join('');

            // Erstelle ein Wrapper-Div für die Items, um die Suche nicht zu beeinträchtigen
            const itemsWrapper = document.createElement('div');
            itemsWrapper.innerHTML = itemsHTML;
            search.insertAdjacentElement('afterend', itemsWrapper);

            // Suchfunktion
            search.addEventListener('input', () => {
                const term = search.value.toLowerCase();
                itemsWrapper.querySelectorAll('label').forEach(item => {
                    const text = item.querySelector('span').textContent.toLowerCase();
                    item.style.display = text.includes(term) ? '' : 'none';
                });
            });
        });
    }

    // --- UI Updates ---
    function updateDropdownText() {
        kpiText.textContent = selectedKpis.length ? selectedKpis.map(k => k.label).join(', ') : 'Keine Kennzahlen';
        groupbyText.textContent = selectedGroups.length ? selectedGroups.map(g => g.label).join(' → ') : 'Keine Gruppierung';
    }

    function updateFilterText(key) {
        const span = document.getElementById(`${key}-selection-text`);
        const count = filters[key].size;
        if (count === 0) {
            span.textContent = 'Alle';
        } else if (count === 1) {
            const value = [...filters[key]][0];
            span.textContent = key === 'match_id' ? formatMatchId(value) : value;
        } else {
            span.textContent = `${count} ausgewählt`;
        }
    }

    // --- Aggregation & Rendering ---
    function updateTable() {
        // Filter anwenden
        const filtered = dataset.filter(row =>
            Object.keys(filters).every(k => filters[k].size === 0 || filters[k].has(row[k]))
        );

        tableHeader.innerHTML = '';
        tableBody.innerHTML = '';

        if (selectedGroups.length === 0 || selectedKpis.length === 0 || filtered.length === 0) {
            tableBody.innerHTML = '<tr><td class="p-4 text-center text-gray-500">Bitte Kennzahlen und Gruppierung auswählen.</td></tr>';
            return;
        }

        // Bevor die Tabelle neu gezeichnet wird, leeren wir nur den Body, nicht den Header
        tableBody.innerHTML = '';

        tableTitle.textContent = `Gruppierung: ${selectedGroups.map(g => g.label).join(' → ')}`;

        // Header
        const headerHtml = `<th class="px-4 py-2 text-left text-xs font-medium text-gray-500">${selectedGroups.map(g => g.label).join(' / ')}</th>`;
        const kpiHeaderHtml = selectedKpis.map(k => `<th class="px-4 py-2 text-right text-xs font-medium text-gray-500">${k.label}</th>`).join('');
        tableHeader.innerHTML = headerHtml + kpiHeaderHtml;

        // Definiere, welche KPIs summiert und welche gemittelt werden sollen
        // Alle KPIs, die auf "_rate", "_accuracy", "_p90", "rating", "possession", "ppda", "field_tilt" enden oder diese enthalten, werden gemittelt.
        // Der Rest wird summiert.
        const AVERAGE_KPIS_PATTERNS = [
            '_rate', '_accuracy', 'rating', 'ppda', 'field_tilt'
        ];
        // Definiere, welche KPIs als Prozentwerte angezeigt werden sollen
        const PERCENTAGE_KPIS_PATTERNS = [
            '_rate', '_accuracy', 'ballbesitz', 'field_tilt'
        ];

        // Definiere KPIs, die immer als ganze Zahlen (Integer) angezeigt werden sollen.
        const INTEGER_KPIS = new Set([
            'goals', 'assists', 'pre_assists', 'sca', 'gca', 'touches',
            'shots', 'shots_on_target', 'shots_off_target', 'shots_blocked',
            'passes_total', 'passes_successful', 'key_passes',
            'progressive_passes', 'passes_into_final_third', 'passes_into_area14', 'third_to_third_passes', 'switches_of_play',
            'dribbles_attempted', 'dribbles_successful', 'progressive_carries', 'box_entries_carry', 'box_entries_total',
            'duels_total', 'duels_won', 'tackles', 'tackles_successful', 'tackles_unsuccessful', 'aerials_total', 'aerials_won',
            'interceptions', 'clearances', 'recoveries', 'pressure_regains', 'dribbled_past', 'defensive_third_touches',
            'saves_total', 'saves_collected', 'saves_parried_safe', 'saves_parried_danger',
            'fouls_committed', 'dispossessed', 'offsides', 'errors',
            'corners_total', 'corners_accurate', 'throw_ins_total', 'throw_ins_accurate',
            'minutes_played'
        ]);
        // Alle p90-Werte sollen ebenfalls summiert werden, da sie bereits normalisiert sind.
        // Raten und prozentuale Werte sind die einzigen, die wirklich gemittelt werden müssen.
        // Aggregation
        const aggTree = {};

        filtered.forEach(row => {
            let currentNode = aggTree;
            selectedGroups.forEach((group, i) => {
                const val = row[group.value] || 'Unbekannt';
                
                // Erstelle den Knoten, falls er nicht existiert. Jeder Knoten hat sum, count und children.
                if (!currentNode[val]) {
                    currentNode[val] = { sum: {}, count: 0, children: {} };
                    selectedKpis.forEach(kpi => {
                        currentNode[val].sum[kpi.value] = 0;
                    });
                }

                // Addiere die Werte auf jeder Ebene hinzu
                selectedKpis.forEach(kpi => {
                    currentNode[val].sum[kpi.value] += (row[kpi.value] || 0);
                });
                // Zähle auf jeder Ebene hoch, um korrekte Durchschnitte für Raten zu ermöglichen
                currentNode[val].count++;
                currentNode = currentNode[val].children;
            });
        });

        // Render
        function render(level, data, path = []) {
            Object.keys(data).forEach(key => {
                const isLeaf = data[key].sum !== undefined;
                const currentPath = path.concat(key); // isLeaf ist jetzt immer true, aber die Logik bleibt
                const rowId = currentPath.join('---');
                const tr = document.createElement('tr');
                
                // Alle Zeilen außer der obersten Ebene sind standardmäßig versteckt
                // Wenn der Parent aufgeklappt ist, zeigen wir die Zeile direkt an
                const isHidden = level > 0 && !expandedRows.has(path.join('---'));
                tr.className = `child-row ${isHidden ? 'hidden' : ''} ${level > 0 ? 'bg-gray-50' : ''}`;
                if (level === 0) tr.classList.add('top-level-row');

                tr.dataset.parent = path.join('---');
                tr.id = `row-${rowId}`;

                

                let cells = '';
                // Eine einzige Gruppierungsspalte mit Einrückung
                const hasChildren = Object.keys(data[key].children).length > 0;
                const indentStyle = `padding-left: ${level * 1.5 + 1}rem;`; // 1rem Basis + 1.5rem pro Ebene
                const chevron = hasChildren ? `<svg class="chevron" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" /></svg>` : `<span class="chevron"></span>`;
                const currentGroupInfo = selectedGroups[level];
                const displayKey = (currentGroupInfo && currentGroupInfo.value === 'match_id') ? formatMatchId(key) : key;
                const groupCellContent = `<div class="flex items-center">${chevron}<span>${displayKey}</span></div>`;
                cells += `<td class="py-2 text-xs whitespace-nowrap" style="${indentStyle}">${groupCellContent}</td>`;

                // Zeige KPIs für JEDE Zeile an
                selectedKpis.forEach(kpi => {
                    let value;
                    const kpiValue = kpi.value;
                    const node = data[key];
                    
                    // Prüfe, ob der KPI gemittelt werden soll.
                    const shouldAverage = AVERAGE_KPIS_PATTERNS.some(pattern => kpiValue.includes(pattern));

                    const isPercentage = PERCENTAGE_KPIS_PATTERNS.some(pattern => kpiValue.includes(pattern));

                    if (shouldAverage) {
                        let avgValue = node.count > 0 ? (node.sum[kpiValue] / node.count).toFixed(2) : '0.00';
                        value = isPercentage ? `${avgValue}%` : avgValue;
                    } else if (INTEGER_KPIS.has(kpiValue)) {
                        value = node.sum[kpiValue].toFixed(0);
                    } else {
                        value = node.sum[kpiValue].toFixed(2);
                    }
                    cells += `<td class="px-4 py-2 text-xs text-right">${value}</td>`;
                });

                tr.innerHTML = cells;
                if (hasChildren) {
                    tr.classList.add('parent-row', 'cursor-pointer');
                    // Wenn die Zeile im Set ist, markieren wir sie als aufgeklappt
                    if (expandedRows.has(rowId)) tr.classList.add('expanded');
                }
                tableBody.appendChild(tr);

                if (Object.keys(data[key].children).length > 0) render(level + 1, data[key].children, currentPath);
            });
        }

        render(0, aggTree);
    }

    // --- Event Listeners ---
    function addEventListeners() {
        // --- Dropdown Toggle (robust für alle Buttons) ---
        document.addEventListener('click', e => {
            // Wenn der Klick innerhalb eines Dropdown-Menüs stattfindet, nichts tun.
            if (e.target.closest('[id$="-checkboxes"]')) {
                return;
            }

            const button = e.target.closest('button[id$="-button"]');
            if (!button) {
                // Klick außerhalb → alle schließen
                document.querySelectorAll('[id$="-checkboxes"]').forEach(el => el.classList.add('hidden'));
                return;
            }

            const id = button.id;
            if (id.includes('-dropdown-button') || id.includes('-filter-button')) {
                e.stopPropagation();
                const targetId = id.replace('-dropdown-button', '-checkboxes').replace('-filter-button', '-checkboxes');
                const target = document.getElementById(targetId);
                if (!target) return;

                const isOpen = !target.classList.contains('hidden');
                // Alle schließen
                document.querySelectorAll('[id$="-checkboxes"]').forEach(el => el.classList.add('hidden'));
                // Dieses öffnen/schließen
                if (isOpen) {
                    target.classList.add('hidden');
                } else {
                    target.classList.remove('hidden');
                }
            }
        });

        // --- KPI Category Toggle ---
        kpiCheckboxes.addEventListener('click', e => {
            const categoryHeader = e.target.closest('.kpi-category > div:first-child');
            if (!categoryHeader) return;

            const parentCategory = categoryHeader.parentElement;
            const itemsDiv = parentCategory.querySelector('.kpi-items');
            itemsDiv.classList.toggle('hidden');
            parentCategory.classList.toggle('expanded');
        });

        // --- KPI Checkboxes ---
        kpiCheckboxes.addEventListener('change', e => {
            if (!e.target.classList.contains('kpi-cb')) return;
            const cb = e.target;
            const item = { value: cb.value, label: cb.dataset.label };
            if (cb.checked) {
                // Entfernen, falls schon da → dann ans Ende
                selectedKpis = selectedKpis.filter(k => k.value !== item.value);
                selectedKpis.push(item);
            } else {
                selectedKpis = selectedKpis.filter(k => k.value !== item.value);
            }
            updateDropdownText();
            updateTable();
        });

        // --- Groupby Checkboxes ---
        groupbyCheckboxes.addEventListener('change', e => {
            if (!e.target.classList.contains('groupby-cb')) return;
            const cb = e.target;
            const item = { value: cb.value, label: cb.dataset.label };
            if (cb.checked) {
                selectedGroups = selectedGroups.filter(g => g.value !== item.value);
                selectedGroups.push(item);
            } else {
                selectedGroups = selectedGroups.filter(g => g.value !== item.value);
            }
            updateDropdownText();
            updateTable();
        });

        // --- Filter Checkboxes ---
        document.querySelectorAll('.filter-cb').forEach(cb => {
            cb.addEventListener('change', () => {
                const key = cb.dataset.key;
                if (cb.checked) {
                    filters[key].add(cb.value);
                }
                else {
                    filters[key].delete(cb.value);
                }
                updateFilterText(key);
                updateTable();
            });
        });

        // --- Clear Buttons ---
        clearKpi.addEventListener('click', () => {
            kpiCheckboxes.querySelectorAll('.kpi-cb').forEach(cb => cb.checked = false);
            selectedKpis = [];
            updateDropdownText();
            updateTable();
        });

        clearGroupby.addEventListener('click', () => {
            groupbyCheckboxes.querySelectorAll('.groupby-cb').forEach(cb => cb.checked = false);
            selectedGroups = [];
            updateDropdownText();
            updateTable();
        });

        ['team', 'opponent', 'player', 'match_id'].forEach(key => {
            document.getElementById(`clear-${key}`).addEventListener('click', () => {
                document.getElementById(`${key}-checkboxes`).querySelectorAll('.filter-cb').forEach(cb => cb.checked = false);
                filters[key].clear();
                updateFilterText(key);
                updateTable();
            });
        });

        // --- Expand/Collapse Rows ---
        tableBody.addEventListener('click', e => {
            const row = e.target.closest('.parent-row');
            if (!row) return;
            const rowId = row.id.replace('row-', '');

            const isExpanding = !row.classList.contains('expanded');

            if (isExpanding) {
                row.classList.add('expanded');
                expandedRows.add(rowId);
                tableBody.querySelectorAll(`[data-parent="${rowId}"]`).forEach(child => child.classList.remove('hidden'));
            } else {
                row.classList.remove('expanded');
                expandedRows.delete(rowId);
                // Schließe alle untergeordneten Zeilen rekursiv
                const descendants = tableBody.querySelectorAll(`[id^="row-${rowId}---"]`);
                descendants.forEach(desc => {
                    desc.classList.add('hidden');
                    desc.classList.remove('expanded');
                    const descId = desc.id.replace('row-', '');
                    expandedRows.delete(descId);
                });
            }
        });
    }

    // --- Start ---
    loadData();
});

(() => {
  const state = { groups: {}, data: [] };

  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    const status = el('#status');
    status.textContent = 'Lade Daten...';
    try {
      const [groups, data] = await Promise.all([
        fetchJSON('kpi_groups.json'),
        fetchJSON('processed_data.json').catch(() => [])
      ]);
      state.groups = groups || {};
      state.data = Array.isArray(data) ? data : [];

      buildGroupSelect(Object.keys(state.groups));
      status.textContent = '';
    } catch (err) {
      console.error(err);
      status.textContent = 'Gruppen konnten nicht geladen werden (kpi_groups.json fehlt?).';
      // Minimaler Fallback: leere Auswahl
      buildGroupSelect([]);
    }
  }

  function buildGroupSelect(keys) {
    const groupSel = el('#groupSelect');
    const metricSel = el('#metricSelect');
    groupSel.innerHTML = '';
    metricSel.innerHTML = '';

    if (!keys || keys.length === 0) {
      groupSel.disabled = true;
      metricSel.disabled = true;
      metricSel.innerHTML = '<option>Keine Kennzahlen verfügbar</option>';
      return;
    }

    // bevorzugte Reihenfolge
    const order = ['Allgemein', 'Spiel mit dem Ball', 'Spiel gg den Ball', 'Standards'];
    const ordered = order.filter(k => keys.includes(k)).concat(keys.filter(k => !order.includes(k)));

    for (const k of ordered) {
      const opt = document.createElement('option');
      opt.value = k;
      opt.textContent = k;
      groupSel.appendChild(opt);
    }

    groupSel.addEventListener('change', () => populateMetrics(groupSel.value));
    const defaultGroup = ordered.includes('Allgemein') ? 'Allgemein' : ordered[0];
    groupSel.value = defaultGroup;
    populateMetrics(defaultGroup);
  }

  function populateMetrics(groupKey) {
    const metricSel = el('#metricSelect');
    metricSel.innerHTML = '';
    const metrics = state.groups[groupKey] || [];

    if (metrics.length === 0) {
      metricSel.disabled = true;
      metricSel.innerHTML = '<option>Keine Kennzahlen in dieser Gruppe</option>';
      return;
    }

    metricSel.disabled = false;
    for (const m of metrics) {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      metricSel.appendChild(opt);
    }
  }

  function el(sel) { return document.querySelector(sel); }
  async function fetchJSON(url) {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
    return res.json();
  }
})();
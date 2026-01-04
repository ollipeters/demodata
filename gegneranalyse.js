document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const teamSelect = document.getElementById('team-select');
    const analysisContent = document.getElementById('analysis-content');
    const initialState = document.getElementById('initial-state');
    const kpiGrid = document.getElementById('kpi-summary-grid');
    const matchesList = document.getElementById('recent-matches-list');

    // --- State ---
    let dataset = [];
    let teams = [];

    // --- KPIs to display in the summary ---
    const SUMMARY_KPIS = [
        { key: 'goals', label: 'Tore', precision: 1 },
        { key: 'xg', label: 'xG', precision: 2 },
        { key: 'shots', label: 'Schüsse', precision: 1 },
        { key: 'possession', label: 'Ballbesitz', precision: 1, suffix: '%' },
        { key: 'pass_accuracy', label: 'Passquote', precision: 1, suffix: '%' },
        { key: 'ppda', label: 'PPDA', precision: 1 },
        { key: 'field_tilt', label: 'Field Tilt', precision: 1, suffix: '%' },
        { key: 'progressive_passes', label: 'Progressive Pässe', precision: 1 },
        { key: 'progressive_carries', label: 'Progressive Läufe', precision: 1 },
        { key: 'box_entries_total', label: 'Strafraum-Aktionen', precision: 1 }
    ];

    // --- Initialization ---
    async function init() {
        try {
            const response = await fetch('processed_data.json');
            if (!response.ok) throw new Error('Daten konnten nicht geladen werden.');
            dataset = await response.json();

            // Get unique team names
            const teamNames = new Set(dataset.map(row => row.team));
            teams = Array.from(teamNames).sort();

            populateTeamSelector();
            addEventListeners();
        } catch (error) {
            console.error("Fehler beim Initialisieren:", error);
            initialState.innerHTML = `<p class="text-red-500">${error.message}</p>`;
        }
    }

    function populateTeamSelector() {
        teams.forEach(team => {
            const option = document.createElement('option');
            option.value = team;
            option.textContent = team;
            teamSelect.appendChild(option);
        });
    }

    function addEventListeners() {
        teamSelect.addEventListener('change', handleTeamSelection);
    }

    // --- Logic ---
    function handleTeamSelection() {
        const selectedTeam = teamSelect.value;

        if (!selectedTeam || selectedTeam === "Bitte ein Team wählen...") {
            analysisContent.classList.add('hidden');
            initialState.classList.remove('hidden');
            return;
        }

        // Filter matches for the selected team
        const teamMatches = dataset.filter(row => row.team === selectedTeam);

        // Get the last 5 unique matches
        const recentMatchIds = [...new Set(teamMatches.map(row => row.match_id))].slice(-5);
        const recentMatchesData = teamMatches.filter(row => recentMatchIds.includes(row.match_id));

        // Calculate summary KPIs
        const summary = calculateSummary(recentMatchesData, selectedTeam);

        // Render the UI
        renderSummary(summary);
        renderRecentMatches(recentMatchIds, selectedTeam);

        analysisContent.classList.remove('hidden');
        initialState.classList.add('hidden');
    }

    function calculateSummary(data, teamName) {
        const summary = {};
        const matchCount = new Set(data.map(d => d.match_id)).size;

        SUMMARY_KPIS.forEach(kpi => {
            // We need to aggregate team-level stats, not player stats.
            // For stats like goals, xg, shots, we sum them up per match first.
            // For stats like ppda, field_tilt, we take the value for the team for each match.
            let totalValue = 0;
            const uniqueMatchValues = new Map();

            data.forEach(row => {
                if (row.team === teamName) {
                    // For team-level stats, we only need one value per match
                    if (['ppda', 'field_tilt', 'possession', 'pass_accuracy'].includes(kpi.key)) {
                        if (!uniqueMatchValues.has(row.match_id)) {
                            uniqueMatchValues.set(row.match_id, row[kpi.key] || 0);
                        }
                    } else {
                        // For player-level stats, sum them up for the team
                        uniqueMatchValues.set(row.match_id, (uniqueMatchValues.get(row.match_id) || 0) + (row[kpi.key] || 0));
                    }
                }
            });

            totalValue = Array.from(uniqueMatchValues.values()).reduce((a, b) => a + b, 0);
            summary[kpi.key] = matchCount > 0 ? totalValue / matchCount : 0;
        });

        return summary;
    }

    // --- Rendering ---
    function renderSummary(summary) {
        kpiGrid.innerHTML = '';
        SUMMARY_KPIS.forEach(kpi => {
            const value = summary[kpi.key] || 0;
            const formattedValue = value.toFixed(kpi.precision);

            const card = `
                <div class="apple-card p-4 text-center">
                    <p class="text-sm text-gray-500">${kpi.label}</p>
                    <p class="text-3xl font-bold stat-value mt-1">${formattedValue}${kpi.suffix || ''}</p>
                </div>
            `;
            kpiGrid.innerHTML += card;
        });
    }

    function renderRecentMatches(matchIds, selectedTeam) {
        matchesList.innerHTML = '';
        matchIds.reverse().forEach(matchId => { // Show most recent first
            const matchData = dataset.find(row => row.match_id === matchId && row.team === selectedTeam);
            if (!matchData) return;

            const opponent = matchData.opponent;
            
            // Find goals for both teams
            const teamGoals = dataset.filter(r => r.match_id === matchId && r.team === selectedTeam).reduce((sum, p) => sum + (p.goals || 0), 0);
            const opponentGoals = dataset.filter(r => r.match_id === matchId && r.team === opponent).reduce((sum, p) => sum + (p.goals || 0), 0);

            let resultBadge = '';
            if (teamGoals > opponentGoals) {
                resultBadge = '<span class="bg-green-100 text-green-800 text-xs font-medium px-2.5 py-0.5 rounded-full">Sieg</span>';
            } else if (teamGoals < opponentGoals) {
                resultBadge = '<span class="bg-red-100 text-red-800 text-xs font-medium px-2.5 py-0.5 rounded-full">Niederlage</span>';
            } else {
                resultBadge = '<span class="bg-yellow-100 text-yellow-800 text-xs font-medium px-2.5 py-0.5 rounded-full">Unentschieden</span>';
            }

            const card = `
                <div class="apple-card p-6 flex items-center justify-between">
                    <div>
                        <p class="text-sm text-gray-500">Gegner</p>
                        <p class="text-xl font-semibold text-gray-800">${opponent}</p>
                    </div>
                    <div class="text-center">
                        <p class="text-3xl font-bold">${teamGoals} - ${opponentGoals}</p>
                    </div>
                    <div class="text-right">
                        ${resultBadge}
                    </div>
                </div>
            `;
            matchesList.innerHTML += card;
        });
    }

    // --- Start the app ---
    init();
});
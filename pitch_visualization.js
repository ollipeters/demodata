// pitch_visualization.js (modular, global API)
(function(global) {
    const LINE_COLOR = '#6b7280';

    function drawPitch(ctx, w, h) {
        ctx.strokeStyle = LINE_COLOR;
        ctx.lineWidth = Math.max(1, w * 0.005);
        ctx.lineCap = 'round';
        ctx.strokeRect(0, 0, w, h);
        ctx.beginPath();
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(w / 2, h / 2, w * 0.2, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(w / 2, h / 2, w * 0.01, 0, 2 * Math.PI);
        ctx.fillStyle = LINE_COLOR;
        ctx.fill();
        const penaltyBoxHeight = h * (16.5 / 105);
        const penaltyBoxWidth = w * (40.3 / 68);
        const penaltyBoxX = (w - penaltyBoxWidth) / 2;
        ctx.strokeRect(penaltyBoxX, 0, penaltyBoxWidth, penaltyBoxHeight);
        ctx.strokeRect(penaltyBoxX, h - penaltyBoxHeight, penaltyBoxWidth, penaltyBoxHeight);
        const goalBoxHeight = h * (5.5 / 105);
        const goalBoxWidth = w * (18.32 / 68);
        const goalBoxX = (w - goalBoxWidth) / 2;
        ctx.strokeRect(goalBoxX, 0, goalBoxWidth, goalBoxHeight);
        ctx.strokeRect(goalBoxX, h - goalBoxHeight, goalBoxWidth, goalBoxHeight);

        ctx.save();
        ctx.strokeStyle = 'rgba(156, 163, 175, 0.35)';
        ctx.lineWidth = Math.max(1, w * 0.002);
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        for (let i = 1; i < 6; i++) {
            ctx.moveTo(0, (h / 6) * i);
            ctx.lineTo(w, (h / 6) * i);
        }
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(w / 3, 0);
        ctx.lineTo(w / 3, h);
        ctx.moveTo(2 * w / 3, 0);
        ctx.lineTo(2 * w / 3, h);
        ctx.stroke();
        ctx.restore();
    }

    function splitCsvLine(line) {
        const out = [];
        let buf = '';
        let bracket = 0;
        for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (ch === '[') bracket++;
            if (ch === ']') bracket = Math.max(0, bracket - 1);
            if (ch === ',' && bracket === 0) {
                out.push(buf);
                buf = '';
            } else {
                buf += ch;
            }
        }
        if (buf.length) out.push(buf);
        return out.map(s => s.replace(/^\"|\"$/g, ''));
    }

    function estimateTopXIFromEvents(allEvents, teamName) {
        const teamEvents = allEvents.filter(e => e.team === teamName && e.playerId != null);
        const counts = new Map();
        teamEvents.forEach(e => {
            const pid = Number(e.playerId);
            counts.set(pid, (counts.get(pid) || 0) + 1);
        });
        const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 11);
        return new Set(sorted.map(([pid]) => pid));
    }

    async function loadStartingXI(formationsPath, teamName, allEvents) {
        try {
            const res = await fetch(formationsPath);
            if (!res.ok) throw new Error('formations.csv konnte nicht geladen werden.');
            const csvText = await res.text();
            const lines = csvText.trim().split(/\r?\n/);
            const header = lines[0].split(',');
            const idxTeam = header.indexOf('team_name');
            const idxStart = header.indexOf('startMinuteExpanded');
            const idxPlayerIds = header.indexOf('playerIds');
            const idxSlots = header.indexOf('formationSlots');
            for (let i = 1; i < lines.length; i++) {
                const parts = splitCsvLine(lines[i]);
                if (!parts.length) continue;
                const team = parts[idxTeam];
                const startMin = parts[idxStart];
                if (team === teamName && Number(startMin) === 0) {
                    const playerIdsArr = JSON.parse(parts[idxPlayerIds]);
                    const slotsArr = JSON.parse(parts[idxSlots]);
                    const ids = new Set();
                    slotsArr.forEach((slot, idx) => {
                        if (Number(slot) >= 1 && Number(slot) <= 11) {
                            ids.add(Number(playerIdsArr[idx]));
                        }
                    });
                    return ids;
                }
            }
            console.warn('Keine Formation (Startminute 0) gefunden – fallback auf Top-11.');
            return estimateTopXIFromEvents(allEvents, teamName);
        } catch (err) {
            console.warn('Formation konnte nicht geladen werden – fallback auf Top-11.', err);
            return estimateTopXIFromEvents(allEvents, teamName);
        }
    }

    function calculateAveragePlayerPositions(teamEvents, allowedIdsSet) {
        const positionEvents = teamEvents.filter(e => e.x_coord != null && e.y_coord != null && e.playerId != null);
        const byPlayer = new Map();
        const nameById = new Map();
        positionEvents.forEach(event => {
            const pid = Number(event.playerId);
            if (allowedIdsSet && !allowedIdsSet.has(pid)) return;
            if (!byPlayer.has(pid)) byPlayer.set(pid, { sum_x: 0, sum_y: 0, count: 0 });
            const agg = byPlayer.get(pid);
            agg.sum_x += Number(event.x_coord);
            agg.sum_y += Number(event.y_coord);
            agg.count++;
            if (event.player_name) nameById.set(pid, event.player_name);
        });
        const result = [];
        for (const [pid, agg] of byPlayer.entries()) {
            if (agg.count === 0) continue;
            result.push({
                player_id: pid,
                player_name: nameById.get(pid) || String(pid),
                avg_x: agg.sum_x / agg.count,
                avg_y: agg.sum_y / agg.count
            });
        }
        if (result.length > 11) {
            const withCount = result.map(p => ({ ...p, count: byPlayer.get(p.player_id).count }));
            withCount.sort((a, b) => b.count - a.count);
            return withCount.slice(0, 11).map(({ count, ...rest }) => rest);
        }
        return result;
    }

    function buildPassNetwork(teamEvents, allowedIdsSet) {
        const events = teamEvents
            .filter(e => e.playerId != null && e.period && typeof e.minute === 'number' && typeof e.second === 'number')
            .map(e => ({
                playerId: Number(e.playerId),
                type: e.event_type,
                outcome: e.outcome,
                period: e.period,
                minute: Number(e.minute),
                second: Number(e.second)
            }));
        const edges = new Map();
        for (let i = 0; i < events.length; i++) {
            const ev = events[i];
            if (ev.type !== 'Pass' || ev.outcome !== 'Successful') continue;
            if (allowedIdsSet && !allowedIdsSet.has(ev.playerId)) continue;
            const t0 = ev.minute * 60 + ev.second;
            const per0 = ev.period;
            let receiverId = null;
            for (let j = i + 1; j < events.length; j++) {
                const nxt = events[j];
                if (nxt.period !== per0) break;
                const dt = (nxt.minute * 60 + nxt.second) - t0;
                if (dt < 0) continue;
                if (dt > 10) break;
                if (nxt.playerId && nxt.playerId !== ev.playerId) { receiverId = nxt.playerId; break; }
            }
            if (!receiverId) continue;
            if (allowedIdsSet && !allowedIdsSet.has(receiverId)) continue;
            const a = Math.min(ev.playerId, receiverId);
            const b = Math.max(ev.playerId, receiverId);
            const key = `${a}-${b}`;
            edges.set(key, (edges.get(key) || 0) + 1);
        }
        return [...edges.entries()].map(([key, count]) => {
            const [a, b] = key.split('-').map(Number);
            return { a_id: a, b_id: b, count };
        });
    }

    function getLastName(fullName) {
        if (!fullName || typeof fullName !== 'string') return '';
        const parts = fullName.trim().split(/\s+/).filter(Boolean);
        return parts.length ? parts[parts.length - 1] : fullName;
    }

    function drawAveragePlayerPositions(ctx, playerPositions, w, h, labelCfg) {
        ctx.globalAlpha = 0.9;
        const fontSize = Math.max(8, w * 0.02);
        ctx.font = `${labelCfg?.weight || 600} ${fontSize}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        for (const player of playerPositions) {
            const x = player.avg_y * (w / 100);
            const y = (100 - player.avg_x) * (h / 100);
            ctx.beginPath();
            ctx.arc(x, y, Math.max(5, w * 0.01), 0, 2 * Math.PI);
            ctx.fillStyle = '#0A3F86';
            ctx.fill();
            const label = (labelCfg?.lastName) ? getLastName(player.player_name) : (player.player_name || '');
            ctx.fillStyle = labelCfg?.color || '#111827';
            ctx.fillText(label, x, y - Math.max(5, w * 0.015));
        }
        ctx.globalAlpha = 1.0;
    }

    function drawPassNetwork(ctx, playerPositions, edges, w, h) {
        if (!edges || !edges.length) return;
        const posById = new Map();
        playerPositions.forEach(p => posById.set(Number(p.player_id), p));
        const counts = edges.map(e => e.count);
        const minC = Math.min(...counts), maxC = Math.max(...counts);
        const minPx = Math.max(1, w * 0.002), maxPx = Math.max(3, w * 0.020);
        const minA = 0.12, maxA = 0.85;
        const gamma = 1.35;
        const widthScale = (c) => {
            if (maxC === minC) return (minPx + maxPx) / 2;
            const t = (c - minC) / (maxC - minC);
            const t2 = Math.pow(Math.max(0, Math.min(1, t)), gamma);
            return minPx + t2 * (maxPx - minPx);
        };
        const alphaScale = (c) => {
            if (maxC === minC) return (minA + maxA) / 2;
            const t = (c - minC) / (maxC - minC);
            const t2 = Math.pow(Math.max(0, Math.min(1, t)), gamma);
            return minA + t2 * (maxA - minA);
        };
        ctx.save();
        ctx.lineCap = 'round';
        edges.forEach(edge => {
            const a = posById.get(Number(edge.a_id));
            const b = posById.get(Number(edge.b_id));
            if (!a || !b) return;
            const ax = a.avg_y * (w / 100);
            const ay = (100 - a.avg_x) * (h / 100);
            const bx = b.avg_y * (w / 100);
            const by = (100 - b.avg_x) * (h / 100);
            ctx.lineWidth = widthScale(edge.count);
            ctx.strokeStyle = `rgba(10, 63, 134, ${alphaScale(edge.count)})`;
            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(bx, by);
            ctx.stroke();
        });
        ctx.restore();
    }

        async function renderAveragePositionsAndNetwork(config) {
        const {
            canvas,
            coordsEl,
            statsEl,
            teamName = 'Hamburg',
                eventsPath = 'events_data.json',
                eventsData = null,
            formationsPath = null,
            passNetwork = { enabled: true },
            label = { lastName: true, color: '#111827', weight: 600 },
        } = config || {};

        if (!canvas) throw new Error('canvas ist erforderlich');
        const parent = canvas.parentElement;

        // Daten laden
            const allEvents = Array.isArray(eventsData) ? eventsData : (await (async () => {
                const r = await fetch(eventsPath);
                if (!r.ok) throw new Error('Events konnten nicht geladen werden');
                return r.json();
            })());
        const teamEvents = allEvents.filter(e => e.team === teamName);

        // Für per-90 der erhaltenen Pässe: Minuten pro Spieler aus processed_data.json summieren (nur Team + relevante Spiele)
        const matchIdsSet = new Set(teamEvents.map(e => e.match_id).filter(Boolean));
        let minutesByPlayerName = new Map();
        try {
            const statsRes = await fetch('processed_data.json');
            if (statsRes.ok) {
                const statsData = await statsRes.json();
                statsData
                    .filter(r => r.team === teamName && (matchIdsSet.size === 0 || matchIdsSet.has(r.match_id)))
                    .forEach(r => {
                        const name = r.player;
                        const mins = Number(r.minutes_played || 0);
                        minutesByPlayerName.set(name, (minutesByPlayerName.get(name) || 0) + mins);
                    });
            } else {
                console.warn('processed_data.json konnte nicht geladen werden (Status). Per-90 für Ballmagneten wird als \"-\" angezeigt.');
            }
        } catch (err) {
            console.warn('processed_data.json konnte nicht geladen werden. Per-90 für Ballmagneten wird als \"-\" angezeigt.', err);
        }

        // Startelf bestimmen
        const xi = formationsPath ? await loadStartingXI(formationsPath, teamName, allEvents)
                                                             : estimateTopXIFromEvents(allEvents, teamName);

        // Aggregationen
        const avgPositions = calculateAveragePlayerPositions(teamEvents, xi);
        const edges = passNetwork?.enabled ? buildPassNetwork(teamEvents, xi) : [];

        // Hilfsfunktion: Ø Pässe pro Ballbesitzphase (bis Ballverlust)
        function computePassesPerPossession(allEvents, teamName) {
            // Gruppiere nach Spiel (falls match_id vorhanden)
            const byMatch = new Map();
            allEvents.forEach((e, idx) => {
                const mid = e.match_id != null ? e.match_id : '__single__';
                if (!byMatch.has(mid)) byMatch.set(mid, []);
                byMatch.get(mid).push({
                    team: e.team,
                    type: e.event_type,
                    period: e.period ?? 0,
                    minute: Number(e.minute ?? 0),
                    second: Number(e.second ?? 0),
                    __i: idx
                });
            });
            const possessions = [];
            for (const [, evs] of byMatch.entries()) {
                evs.sort((a, b) => (a.period - b.period) || (a.minute - b.minute) || (a.second - b.second) || (a.__i - b.__i));
                let currentTeam = null;
                let ourPasses = 0;
                let inOurPossession = false;
                for (const ev of evs) {
                    if (currentTeam === null) {
                        currentTeam = ev.team;
                        inOurPossession = (currentTeam === teamName);
                        ourPasses = 0;
                    } else if (ev.team !== currentTeam) {
                        // Teamwechsel => alte Ballbesitzphase endet
                        if (inOurPossession) possessions.push(ourPasses);
                        currentTeam = ev.team;
                        inOurPossession = (currentTeam === teamName);
                        ourPasses = 0;
                    }
                    if (inOurPossession && ev.team === teamName && ev.type === 'Pass') {
                        ourPasses += 1;
                    }
                }
                // Match-Ende: letzte Phase abschließen
                if (inOurPossession) possessions.push(ourPasses);
            }
            if (!possessions.length) return null;
            const avg = possessions.reduce((a, b) => a + b, 0) / possessions.length;
            return Math.round(avg * 10) / 10;
        }

        const passesPerPoss = computePassesPerPossession(allEvents, teamName);

        function drawAll() {
            canvas.width = parent.clientWidth;
            canvas.height = parent.clientHeight;
            const ctx = canvas.getContext('2d');
            drawPitch(ctx, canvas.width, canvas.height);
            if (passNetwork?.enabled) drawPassNetwork(ctx, avgPositions, edges, canvas.width, canvas.height);
            drawAveragePlayerPositions(ctx, avgPositions, canvas.width, canvas.height, label);
            if (statsEl) {
                // KPIs fürs Passnetz: Pässe aus 1. und 2. Drittel + Erfolgsquoten
                const allPasses = teamEvents.filter(e => e.event_type === 'Pass' && e.x_coord != null && e.end_x != null);
                // 1. Drittel (Start im 1. Drittel)
                const firstAll = allPasses.filter(e => Number(e.x_coord) < 33.3);
                const firstSucc = firstAll.filter(e => e.outcome === 'Successful');
                const firstCount = firstAll.length;
                const firstRate = firstCount ? Math.round((firstSucc.length / firstCount) * 100) : 0;
                // 2. Drittel (Start im 2. Drittel)
                const secondAll = allPasses.filter(e => Number(e.x_coord) >= 33.3 && Number(e.x_coord) < 66.6);
                const secondSucc = secondAll.filter(e => e.outcome === 'Successful');
                const secondCount = secondAll.length;
                const secondRate = secondCount ? Math.round((secondSucc.length / secondCount) * 100) : 0;

                // Top 3 Empfänger (nur erfolgreiche Pässe, gleiche Logik wie beim Passnetz, optional gefiltert auf XI)
                const seqEvents = teamEvents
                    .filter(e => e.playerId != null && e.period && typeof e.minute === 'number' && typeof e.second === 'number')
                    .map(e => ({
                        playerId: Number(e.playerId),
                        type: e.event_type,
                        outcome: e.outcome,
                        period: e.period,
                        minute: Number(e.minute),
                        second: Number(e.second),
                        player_name: e.player_name || ''
                    }));
                const nameById = new Map();
                seqEvents.forEach(e => { if (e.playerId) nameById.set(e.playerId, e.player_name || String(e.playerId)); });
                const recvCount = new Map();
                for (let i = 0; i < seqEvents.length; i++) {
                    const ev = seqEvents[i];
                    if (ev.type !== 'Pass' || ev.outcome !== 'Successful') continue;
                    if (xi && !xi.has(ev.playerId)) continue; // optional: Absender im XI
                    const t0 = ev.minute * 60 + ev.second;
                    const per0 = ev.period;
                    let receiverId = null;
                    for (let j = i + 1; j < seqEvents.length; j++) {
                        const nxt = seqEvents[j];
                        if (nxt.period !== per0) break;
                        const dt = (nxt.minute * 60 + nxt.second) - t0;
                        if (dt < 0) continue;
                        if (dt > 10) break;
                        if (nxt.playerId && nxt.playerId !== ev.playerId) { receiverId = nxt.playerId; break; }
                    }
                    if (!receiverId) continue;
                    if (xi && !xi.has(receiverId)) continue; // optional: Empfänger im XI
                    recvCount.set(receiverId, (recvCount.get(receiverId) || 0) + 1);
                }
                const topReceivers = [...recvCount.entries()]
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5)
                    .map(([pid, cnt]) => {
                        const full = nameById.get(pid) || String(pid);
                        return { name: getLastName(full), fullName: full, count: cnt };
                    });
                // Denominator for percentages: all team passes
                const denomAllPasses = teamEvents.filter(e => e.event_type === 'Pass').length;
                // Total minutes across matches for per-90 scaling
                const totalMinutes = (() => {
                    // Group by match_id if present; else use a single bucket
                    const byMatch = new Map();
                    allEvents.forEach(ev => {
                        const mid = ev.match_id != null ? ev.match_id : '__single__';
                        if (!byMatch.has(mid)) byMatch.set(mid, []);
                        byMatch.get(mid).push(ev);
                    });
                    let minutes = 0;
                    for (const [, evs] of byMatch.entries()) {
                        let maxSec = 0;
                        for (const ev of evs) {
                            const per = Number(ev.period ?? 1);
                            const base = (Math.max(0, per - 1)) * 45 * 60; // 45 min per period baseline
                            const m = Number(ev.minute ?? 0);
                            const s = Number(ev.second ?? 0);
                            const t = base + m * 60 + s;
                            if (t > maxSec) maxSec = t;
                        }
                        minutes += maxSec / 60;
                    }
                    return Math.round(minutes * 10) / 10;
                })();
                const pct = (cnt) => (!denomAllPasses ? '-' : `${Math.round((cnt / denomAllPasses) * 100)}%`);
                const per90FromMinutes = (fullName, cnt) => {
                    const mins = minutesByPlayerName.get(fullName) || 0;
                    if (!mins) return '-';
                    return `${Math.round((cnt / mins) * 90)}`;
                };

                // PPDA: Gegnerische Pässe im Aufbaubereich (x<60) / unsere Defensivaktionen dort
                const oppPassesPress = allEvents
                    .filter(e => e.team !== teamName && e.event_type === 'Pass' && e.x_coord != null && Number(e.x_coord) < 60).length;
                const ourDefActions = teamEvents
                    .filter(e => ['Tackle','Interception','BallRecovery','Foul'].includes(e.event_type) && e.x_coord != null && Number(e.x_coord) < 60).length;
                const ppda = ourDefActions ? Math.round((oppPassesPress / ourDefActions) * 10) / 10 : null;

                statsEl.innerHTML = `
                    <div class=\"flex justify-between\"><span>Spieler:</span><span class=\"font-medium text-gray-900\">${avgPositions.length}</span></div>
                    <div class=\"flex justify-between\"><span>Pässe aus dem 1. Drittel:</span><span class=\"font-medium text-gray-900\">${firstCount}</span></div>
                    <div class=\"flex justify-between\"><span>Erfolgsquote 1. Drittel:</span><span class=\"font-medium text-gray-900\">${firstRate}%</span></div>
                    <div class=\"flex justify-between\"><span>Pässe aus dem 2. Drittel:</span><span class=\"font-medium text-gray-900\">${secondCount}</span></div>
                    <div class=\"flex justify-between\"><span>Erfolgsquote 2. Drittel:</span><span class=\"font-medium text-gray-900\">${secondRate}%</span></div>
                    <div class=\"flex justify-between\"><span>Ø Pässe/Phase bis Ballverlust:</span><span class=\"font-medium text-gray-900\">${passesPerPoss != null ? passesPerPoss : '-'}</span></div>
                    <div class=\"flex justify-between\"><span>PPDA (x<60):</span><span class=\"font-medium text-gray-900\">${ppda != null ? ppda : '-'}${ppda != null ? '' : ''}</span></div>
                    ${topReceivers.length ? `
                    <div class=\"mt-3 pt-2 border-t border-gray-200 text-xs text-gray-600\">Ballmagneten</div>
                    ${topReceivers[0] ? `<div>1. <span class=\"font-medium text-gray-900\">${topReceivers[0].name}</span> (${per90FromMinutes(topReceivers[0].fullName, topReceivers[0].count)} p90 | ${pct(topReceivers[0].count)})</div>` : ''}
                    ${topReceivers[1] ? `<div>2. <span class=\"font-medium text-gray-900\">${topReceivers[1].name}</span> (${per90FromMinutes(topReceivers[1].fullName, topReceivers[1].count)} p90 | ${pct(topReceivers[1].count)})</div>` : ''}
                    ${topReceivers[2] ? `<div>3. <span class=\"font-medium text-gray-900\">${topReceivers[2].name}</span> (${per90FromMinutes(topReceivers[2].fullName, topReceivers[2].count)} p90 | ${pct(topReceivers[2].count)})</div>` : ''}
                    ${topReceivers[3] ? `<div>4. <span class=\"font-medium text-gray-900\">${topReceivers[3].name}</span> (${per90FromMinutes(topReceivers[3].fullName, topReceivers[3].count)} p90 | ${pct(topReceivers[3].count)})</div>` : ''}
                    ${topReceivers[4] ? `<div>5. <span class=\"font-medium text-gray-900\">${topReceivers[4].name}</span> (${per90FromMinutes(topReceivers[4].fullName, topReceivers[4].count)} p90 | ${pct(topReceivers[4].count)})</div>` : ''}
                    ` : ''}
                `;
            }
        }

        drawAll();
        const ro = new ResizeObserver(drawAll);
        ro.observe(parent);
        if (coordsEl) {
            canvas.addEventListener('mousemove', (e) => {
                const rect = canvas.getBoundingClientRect();
                const x_coord = ((1 - (e.clientY - rect.top) / canvas.height) * 100).toFixed(1);
                const y_coord = ((e.clientX - rect.left) / canvas.width * 100).toFixed(1);
                coordsEl.textContent = `x: ${x_coord}, y: ${y_coord}`;
            });
            canvas.addEventListener('mouseleave', () => { coordsEl.textContent = ''; });
        }
        return { destroy: () => { ro.disconnect(); } };
    }

    // Zeichnet Abstöße des Torwarts (Linien + Endmarker), Bottom->Top Ausrichtung
    function drawGoalKicks(ctx, kicks, w, h) {
        ctx.save();
        ctx.lineCap = 'round';
        const uniformW = Math.max(1.8, w * 0.007);
        const ah = Math.max(6, w * 0.012);
        const aw = Math.max(3, w * 0.006);
        kicks.forEach(k => {
            const sx = Number(k.y_coord) * (w / 100);
            const sy = (100 - Number(k.x_coord)) * (h / 100);
            const ex = Number(k.end_y) * (w / 100);
            const ey = (100 - Number(k.end_x)) * (h / 100);
            const success = k.outcome === 'Successful';
            const color = success ? 'rgba(10, 63, 134, 0.85)' : 'rgba(255, 107, 107, 0.85)';
            ctx.lineWidth = uniformW;
            ctx.strokeStyle = color;
            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.lineTo(ex, ey);
            ctx.stroke();

            // Arrowhead (wie bei progressiven Pässen)
            const angle = Math.atan2(ey - sy, ex - sx);
            ctx.beginPath();
            ctx.moveTo(ex, ey);
            ctx.lineTo(ex - ah * Math.cos(angle - Math.PI / 6), ey - ah * Math.sin(angle - Math.PI / 6));
            ctx.lineTo(ex - ah * Math.cos(angle + Math.PI / 6), ey - ah * Math.sin(angle + Math.PI / 6));
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();
        });
        ctx.restore();
    }

        async function renderGoalKicksMap(config) {
            const {
                canvas,
                coordsEl,
                statsEl,
                teamName = 'Hamburg',
                eventsPath = 'events_data.json',
                eventsData = null,
                includeOnlyGK = true, // in der Regel ist_goalkick reicht, GK ist Absender
                showAvgLength = true,
            } = config || {};

            if (!canvas) throw new Error('canvas ist erforderlich');
            const parent = canvas.parentElement;

            // Events laden/übernehmen
            const allEvents = Array.isArray(eventsData) ? eventsData : (await (async () => {
                const r = await fetch(eventsPath);
                if (!r.ok) throw new Error('Events konnten nicht geladen werden');
                return r.json();
            })());

            // Filtern: Team + is_goalkick
            const teamEvents = allEvents.filter(e => e.team === teamName);
            const goalKicks = teamEvents.filter(e => e.is_goalkick === true && e.x_coord != null && e.y_coord != null && e.end_x != null && e.end_y != null);

            const matchIds = new Set(teamEvents.map(e => String(e.match_id ?? '')).filter(Boolean));
            const gamesCount = matchIds.size || 1;

            const fmtP90 = (v) => {
                const n = Number(v);
                if (!Number.isFinite(n)) return '-';
                return (Math.abs(n - Math.round(n)) < 1e-9) ? String(Math.round(n)) : String(n);
            };

            // Stats
            const total = goalKicks.length;
            const success = goalKicks.filter(k => k.outcome === 'Successful').length;
            const successRate = total ? Math.round((success / total) * 100) : 0;
            const totalPer90 = Math.round((total / gamesCount) * 10) / 10;
            const successPer90 = Math.round((success / gamesCount) * 10) / 10;
            const avgLen = (() => {
                const list = goalKicks.map(k => (k.pass_length != null ? Number(k.pass_length) : null)).filter(v => v != null);
                if (list.length === 0) return null;
                const s = list.reduce((a, b) => a + b, 0) / list.length;
                return Math.round(s * 10) / 10;
            })();

            function drawAll() {
                canvas.width = parent.clientWidth;
                canvas.height = parent.clientHeight;
                const ctx = canvas.getContext('2d');
                drawPitch(ctx, canvas.width, canvas.height);
                drawGoalKicks(ctx, goalKicks, canvas.width, canvas.height);
                if (statsEl) {
                    const parts = [
                        `<div class=\"flex justify-between\"><span>Abstöße p90 | davon erfolgreich</span><span class=\"font-medium text-gray-900\">${fmtP90(totalPer90)} | ${fmtP90(successPer90)}</span></div>`,
                        `<div class=\"flex justify-between\"><span>Erfolgsquote:</span><span class=\"font-medium text-gray-900\">${successRate}%</span></div>`
                    ];
                    if (showAvgLength && avgLen != null) {
                        parts.push(`<div class=\"flex justify-between\"><span>Ø Länge:</span><span class=\"font-medium text-gray-900\">${avgLen}m</span></div>`);
                    }
                    statsEl.innerHTML = parts.join('');
                }
            }

            drawAll();
            const ro = new ResizeObserver(drawAll);
            ro.observe(parent);
            if (coordsEl) {
                canvas.addEventListener('mousemove', (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const x_coord = ((1 - (e.clientY - rect.top) / canvas.height) * 100).toFixed(1);
                    const y_coord = ((e.clientX - rect.left) / canvas.width * 100).toFixed(1);
                    coordsEl.textContent = `x: ${x_coord}, y: ${y_coord}`;
                });
                canvas.addEventListener('mouseleave', () => { coordsEl.textContent = ''; });
            }

            return { destroy: () => { ro.disconnect(); } };
        }

        // Progressive Pässe
        function metersDistance(xu, yu, xv, yv) {
            // xu/xv in [0,100] along pitch length (105m); yu/yv in [0,100] along width (68m)
            const dxm = ((xv - xu) / 100) * 105;
            const dym = ((yv - yu) / 100) * 68;
            return Math.hypot(dxm, dym);
        }

        function computeProgressivePasses(allEvents, teamName, { minGainM = 10, minLengthM = 5 } = {}) {
            const teamEvents = allEvents.filter(e => e.team === teamName);
            const passes = teamEvents.filter(e => e.event_type === 'Pass' && e.x_coord != null && e.y_coord != null && e.end_x != null && e.end_y != null);
            const out = [];
            for (const p of passes) {
                // Work in raw event space for logic: top goal at (0, 50) in data coordinates
                const xs_raw = Number(p.x_coord);
                const ys_raw = Number(p.y_coord);
                const xe_raw = Number(p.end_x);
                const ye_raw = Number(p.end_y);
                const lengthM = metersDistance(xs_raw, ys_raw, xe_raw, ye_raw);
                const startToGoal = metersDistance(xs_raw, ys_raw, 0, 50);
                const endToGoal = metersDistance(xe_raw, ye_raw, 0, 50);
                const gainM = startToGoal - endToGoal;
                if (lengthM >= minLengthM && gainM >= minGainM) {
                    out.push({
                        xs_raw, ys_raw, xe_raw, ye_raw,
                        gainM: Math.round(gainM * 10) / 10,
                        lengthM: Math.round(lengthM * 10) / 10,
                        player_name: p.player_name || '',
                        outcome: p.outcome
                    });
                }
            }
            return out;
        }

        function drawProgressivePasses(ctx, prog, w, h, orientation = 'bottom-to-top') {
            if (!prog || !prog.length) return;
            ctx.save();
            ctx.lineCap = 'round';
            const uniformW = Math.max(1.8, w * 0.007);
            prog.forEach(p => {
            const isTopToBottom = orientation === 'top-to-bottom';
                // bottom-to-top: team attacks bottom->top visually (progressive passes go upwards)
                // top-to-bottom: 180° rotation incl. left-right mirror
                const sx = (isTopToBottom ? (100 - p.ys_raw) : p.ys_raw) * (w / 100);
                const sy = (isTopToBottom ? (100 - p.xs_raw) : p.xs_raw) * (h / 100);
                const ex = (isTopToBottom ? (100 - p.ye_raw) : p.ye_raw) * (w / 100);
                const ey = (isTopToBottom ? (100 - p.xe_raw) : p.xe_raw) * (h / 100);
                ctx.lineWidth = uniformW;
                const success = p.outcome === 'Successful';
                // Erfolgreich: HSV Blau, sonst Orange-Rot
                ctx.strokeStyle = success ? 'rgba(10, 63, 134, 0.85)' : 'rgba(255, 107, 107, 0.85)';
                ctx.beginPath();
                ctx.moveTo(sx, sy);
                ctx.lineTo(ex, ey);
                ctx.stroke();
                // Arrowhead
                const angle = Math.atan2(ey - sy, ex - sx);
                const ah = Math.max(6, w * 0.012);
                const aw = Math.max(3, w * 0.006);
                ctx.beginPath();
                ctx.moveTo(ex, ey);
                ctx.lineTo(ex - ah * Math.cos(angle - Math.PI / 6), ey - ah * Math.sin(angle - Math.PI / 6));
                ctx.lineTo(ex - ah * Math.cos(angle + Math.PI / 6), ey - ah * Math.sin(angle + Math.PI / 6));
                ctx.closePath();
                ctx.fillStyle = success ? 'rgba(10, 63, 134, 0.85)' : 'rgba(255, 107, 107, 0.85)';
                ctx.fill();
            });
            ctx.restore();
        }

        function buildZoneCountsFromProgressive(prog) {
            // 3 columns (y: 0-33.33, 33.33-66.66, 66.66-100) x 6 rows (x: 0..100 in 6 equal bands)
            const rows = 6, cols = 3;
            const counts = Array.from({ length: rows }, () => Array(cols).fill(0));
            let maxCount = 0;
            const rowBand = 100 / rows; // in data x space (top=0 -> bottom=100)
            const colBand = 100 / cols; // in data y space (left=0 -> right=100)
            for (const p of prog) {
                // use start location of the pass (raw data coords)
                let r = Math.floor(p.xs_raw / rowBand);
                let c = Math.floor(p.ys_raw / colBand);
                if (r < 0) r = 0; if (r >= rows) r = rows - 1;
                if (c < 0) c = 0; if (c >= cols) c = cols - 1;
                counts[r][c]++;
                if (counts[r][c] > maxCount) maxCount = counts[r][c];
            }
            return { counts, maxCount };
        }

        function drawZoneHeat(ctx, counts, maxCount, w, h, orientation = 'bottom-to-top') {
            const rows = counts.length;
            const cols = counts[0]?.length || 0;
            if (!rows || !cols) return;
            const cellH = h / rows;
            const cellW = w / cols;
            const minA = 0.08, maxA = 0.8, gamma = 1.2;
            const isTopToBottom = orientation === 'top-to-bottom';
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const v = counts[r][c];
                    if (v <= 0) continue;
                    const t = maxCount > 0 ? Math.pow(v / maxCount, gamma) : 0;
                    const alpha = minA + t * (maxA - minA);
                    ctx.fillStyle = `rgba(10, 63, 134, ${alpha})`;
                    const x = c * cellW;
                    const y = isTopToBottom ? ((rows - 1 - r) * cellH) : (r * cellH);
                    ctx.fillRect(x, y, cellW, cellH);
                }
            }
        }

        async function renderProgressivePassesMap(config) {
            const {
                canvas,
                coordsEl,
                statsEl,
                teamName = 'Hamburg',
                eventsPath = 'events_data.json',
                eventsData = null,
                minGainM = 10,
                minLengthM = 5,
                viewMode = 'zones', // 'zones' | 'arrows'
                orientation = 'bottom-to-top', // 'bottom-to-top' | 'top-to-bottom'
            } = config || {};

            if (!canvas) throw new Error('canvas ist erforderlich');
            const parent = canvas.parentElement;

            const allEvents = Array.isArray(eventsData) ? eventsData : (await (async () => {
                const r = await fetch(eventsPath);
                if (!r.ok) throw new Error('Events konnten nicht geladen werden');
                return r.json();
            })());

            const matchIds = new Set((allEvents || []).map(e => String(e.match_id ?? '')).filter(Boolean));
            const gamesCount = matchIds.size || 1;
            const fmtP90 = (v) => {
                const n = Number(v);
                if (!Number.isFinite(n)) return '-';
                return (Math.abs(n - Math.round(n)) < 1e-9) ? String(Math.round(n)) : String(n);
            };

            const prog = computeProgressivePasses(allEvents, teamName, { minGainM, minLengthM });
            const succCount = prog.filter(p => p.outcome === 'Successful').length;
            const failCount = prog.length - succCount;
            const { counts, maxCount } = buildZoneCountsFromProgressive(prog);

            // Zonen-Kennzahlen (Start/Ende) für 18-Zonen-Raster wie in Gegneranalyse
            function mapTo18ZoneFromRaw(x, y) {
                if (x == null || y == null) return null;
                const nx = Math.max(0, Math.min(99.999, Number(x)));
                const ny = Math.max(0, Math.min(99.999, Number(y)));
                const rowFromBottom = Math.floor((nx / 100) * 6); // 0 (unten) .. 5 (oben)
                const col = Math.floor((ny / 100) * 3);          // 0 (links) .. 2 (rechts)
                const rowFromTop = 5 - rowFromBottom;
                return rowFromTop * 3 + col + 1; // 1 unten links, 18 oben rechts
            }

            const startZoneCounts = new Map();
            const endZoneCounts = new Map();
            prog.forEach(p => {
                const sz = mapTo18ZoneFromRaw(p.xs_raw, p.ys_raw);
                const ez = mapTo18ZoneFromRaw(p.xe_raw, p.ye_raw);
                if (sz != null) startZoneCounts.set(sz, (startZoneCounts.get(sz) || 0) + 1);
                if (ez != null) endZoneCounts.set(ez, (endZoneCounts.get(ez) || 0) + 1);
            });

            function getMaxZone(map) {
                if (!map.size) return { zone: '-', count: 0 };
                let maxZone = null;
                let maxVal = -1;
                for (const [z, c] of map.entries()) {
                    if (c > maxVal) { maxVal = c; maxZone = z; }
                }
                return { zone: maxZone, count: maxVal };
            }

            const maxStart = getMaxZone(startZoneCounts);
            const maxEnd = getMaxZone(endZoneCounts);

            let currentMode = viewMode === 'arrows' ? 'arrows' : 'zones';

            function drawAll() {
                canvas.width = parent.clientWidth;
                canvas.height = parent.clientHeight;
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                if (currentMode === 'zones') {
                    // Zone heat first, then pitch markings on top for clarity
                    drawZoneHeat(ctx, counts, maxCount, canvas.width, canvas.height, orientation);
                    drawPitch(ctx, canvas.width, canvas.height);
                } else {
                    // Arrows: draw pitch then arrows on top
                    drawPitch(ctx, canvas.width, canvas.height);
                    drawProgressivePasses(ctx, prog, canvas.width, canvas.height, orientation);
                }
                if (statsEl) {
                    const total = prog.length;
                    const avgGain = total ? Math.round((prog.reduce((a, b) => a + b.gainM, 0) / total) * 10) / 10 : 0;
                    const totalPer90 = Math.round((total / gamesCount) * 10) / 10;
                    const succPer90 = Math.round((succCount / gamesCount) * 10) / 10;
                    const failPer90 = Math.round((failCount / gamesCount) * 10) / 10;
                    const maxStartPer90 = Math.round((maxStart.count / gamesCount) * 10) / 10;
                    const maxEndPer90 = Math.round((maxEnd.count / gamesCount) * 10) / 10;
                    const maxCountPer90 = Math.round((maxCount / gamesCount) * 10) / 10;
                    const base = [
                        `<div class=\"flex justify-between\"><span>Progressive Pässe p90:</span><span class=\"font-medium text-gray-900\">${fmtP90(totalPer90)}</span></div>`,
                        `<div class=\"flex justify-between\"><span>Ø Vorwärtsgewinn:</span><span class=\"font-medium text-gray-900\">${avgGain} m</span></div>`,
                        `<div class=\"flex justify-between\"><span>Meisten Pässe aus Zone:</span><span class=\"font-medium text-gray-900\">${maxStart.zone} (${fmtP90(maxStartPer90)})</span></div>`,
                        `<div class=\"flex justify-between\"><span>Meisten Pässe in Zone:</span><span class=\"font-medium text-gray-900\">${maxEnd.zone} (${fmtP90(maxEndPer90)})</span></div>`
                    ];
                    const extra = currentMode === 'zones'
                        ? [`<div class=\"flex justify-between\"><span>Max in Zone p90:</span><span class=\"font-medium text-gray-900\">${fmtP90(maxCountPer90)}</span></div>`]
                        : [
                            `<div class=\"flex justify-between\"><span>Erfolgreich p90:</span><span class=\"font-medium text-gray-900\">${fmtP90(succPer90)}</span></div>`,
                            `<div class=\"flex justify-between\"><span>Nicht erfolgreich p90:</span><span class=\"font-medium text-gray-900\">${fmtP90(failPer90)}</span></div>`
                          ];
                    statsEl.innerHTML = [...base, ...extra].join('');
                }
            }

            drawAll();
            const ro = new ResizeObserver(drawAll);
            ro.observe(parent);
            if (coordsEl) {
                canvas.addEventListener('mousemove', (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const isTopToBottom = orientation === 'top-to-bottom';
                    const relY = (e.clientY - rect.top) / canvas.height;
                    const relX = (e.clientX - rect.left) / canvas.width;
                    const x_coord = ((isTopToBottom ? (1 - relY) : relY) * 100).toFixed(1);
                    const y_coord = ((isTopToBottom ? (1 - relX) : relX) * 100).toFixed(1);
                    coordsEl.textContent = `x: ${x_coord}, y: ${y_coord}`;
                });
                canvas.addEventListener('mouseleave', () => { coordsEl.textContent = ''; });
            }

            return {
                destroy: () => { ro.disconnect(); },
                setViewMode: (mode) => {
                    currentMode = (mode === 'arrows') ? 'arrows' : 'zones';
                    drawAll();
                }
            };
        }

        global.PitchViz = { renderAveragePositionsAndNetwork, renderGoalKicksMap, renderProgressivePassesMap };
})(window);
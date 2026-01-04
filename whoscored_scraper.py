import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError, sync_playwright

# Set logging level for playwright
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

WHO_SCORED_BASE_URL = "https://www.whoscored.com"

DEFAULT_TEAM_FIXTURES: Dict[str, str] = {
    "bayer-leverkusen": "https://www.whoscored.com/teams/36/fixtures/germany-bayer-leverkusen",
    "hamburger-sv": "https://www.whoscored.com/teams/38/fixtures/germany-hamburger-sv",
    "freiburg": "https://www.whoscored.com/teams/50/fixtures/germany-freiburg",
}


@dataclass(frozen=True)
class ScrapeOptions:
    headless: bool = True
    show_more_clicks: int = 20
    per_team_limit: Optional[int] = None
    delay_seconds: float = 1.0
    force: bool = False
    competition_slug: str = "germany-bundesliga"
    season_slug: str = "2025-2026"

def summarize_stats(stats_dict):
    """
    Summarizes the minute-by-minute stats into totals.
    e.g., {"10": 1, "25": 2} -> 3
    """
    totals = {}
    if not stats_dict:
        return totals
        
    for key, value in stats_dict.items():
        if isinstance(value, dict) and key == 'ratings':
            # Get the last rating as the final rating
            try:
                totals['final_rating'] = list(value.values())[-1]
            except IndexError:
                totals['final_rating'] = None
        elif isinstance(value, dict):
            # Sum all values in the dict for totals
            totals[key] = sum(value.values())
    if 'totalMinutesPlayed' in stats_dict:
        totals['minutes_played'] = stats_dict['totalMinutesPlayed']
    return totals

def parse_game_info(data):
    """Extracts top-level match metadata."""
    log.info("Verarbeite Spiel-Metadaten...")
    info: Dict[str, Any] = {
        'matchId': data.get('matchId'),
        'attendance': data.get('attendance'),
        'venueName': data.get('venueName'),
        'startTime': data.get('startTime'),
        'score': data.get('score'),
        'htScore': data.get('htScore'),
        'ftScore': data.get('ftScore'),
        'referee_name': data.get('referee', {}).get('name'),
        'home_team_id': data.get('home', {}).get('teamId'),
        'home_team_name': data.get('home', {}).get('name'),
        'away_team_id': data.get('away', {}).get('teamId'),
        'away_team_name': data.get('away', {}).get('name'),
    }
    return info

def parse_team_stats(data):
    """Extracts and summarizes home and away team stats."""
    log.info("Verarbeite Team-Statistiken...")
    teams_stats: List[Dict[str, Any]] = []
    
    home_stats = summarize_stats(data.get('home', {}).get('stats', {}))
    home_stats['team_name'] = data.get('home', {}).get('name')
    home_stats['field'] = 'home'
    
    away_stats = summarize_stats(data.get('away', {}).get('stats', {}))
    away_stats['team_name'] = data.get('away', {}).get('name')
    away_stats['field'] = 'away'
    
    teams_stats.append(home_stats)
    teams_stats.append(away_stats)
    
    return teams_stats

def parse_player_stats(data):
    """Extracts and summarizes stats for every player."""
    log.info("Verarbeite Spieler-Statistiken...")
    all_players_data: List[Dict[str, Any]] = []
    
    home_players = data.get('home', {}).get('players', [])
    away_players = data.get('away', {}).get('players', [])
    
    for player in home_players + away_players:
        player_info = {
            'playerId': player.get('playerId'),
            'shirtNo': player.get('shirtNo'),
            'name': player.get('name'),
            'position': player.get('position'),
            'team_name': data.get(player.get('field'), {}).get('name'),
            'field': player.get('field'),
            'age': player.get('age'),
            'height': player.get('height'),
            'weight': player.get('weight'),
            'isFirstEleven': player.get('isFirstEleven'),
            'minutes_played': player.get('minutesPlayed'),
            'isManOfTheMatch': player.get('isManOfTheMatch'),
        }
        
        # Summarize the player's minute-by-minute stats
        player_stats = summarize_stats(player.get('stats', {}))
        player_info.update(player_stats)
        all_players_data.append(player_info)
        
    return all_players_data

def parse_events(data):
    """Extracts all match events, including detailed qualifiers."""
    log.info("Verarbeite Spiel-Events (inkl. Qualifiers)...")
    events = data.get('incidentEvents', data.get('events', []))
    if not events:
        log.warning("Keine 'incidentEvents' oder 'events' im Datenobjekt gefunden.")
        return []

    player_dict = data.get('playerIdNameDictionary', {})
    team_dict = {
        data.get('home', {}).get('teamId'): data.get('home', {}).get('name', 'Home'),
        data.get('away', {}).get('teamId'): data.get('away', {}).get('name', 'Away')
    }
    
    processed_events: List[Dict[str, Any]] = []
    for event in events:
        # --- NEU: Strukturierte Verarbeitung der Qualifiers ---
        qualifiers = event.get('qualifiers', [])
        q_data: Dict[str, Any] = {
            'pass_end_x': None, 'pass_end_y': None, 'pass_angle': None, 'pass_length': None,
            'is_longball': False, 'is_chipped': False, 'is_cross': False, 'is_throughball': False,
            'is_headpass': False, 'is_freekick': False, 'is_corner': False, 'is_throwin': False,
            'is_header': False, 'is_penalty': False, 'is_goalkick': False
        }
        for q in qualifiers:
            q_type = q.get('type', {}).get('displayName')
            if q_type == 'PassEndX': q_data['pass_end_x'] = q.get('value')

            elif q_type == 'PassEndY': q_data['pass_end_y'] = q.get('value')
            elif q_type == 'Angle': q_data['pass_angle'] = q.get('value')
            elif q_type == 'Length': q_data['pass_length'] = q.get('value')
            elif q_type == 'Longball': q_data['is_longball'] = True
            elif q_type == 'Chipped': q_data['is_chipped'] = True
            elif q_type == 'Cross': q_data['is_cross'] = True
            elif q_type == 'Throughball': q_data['is_throughball'] = True
            elif q_type == 'HeadPass': q_data['is_headpass'] = True
            elif q_type == 'FreekickTaken': q_data['is_freekick'] = True
            elif q_type == 'CornerTaken': q_data['is_corner'] = True
            elif q_type == 'ThrowIn': q_data['is_throwin'] = True
            elif q_type == 'Head': q_data['is_header'] = True
            elif q_type == 'GoalKick' or q_type == 'FromGoalKick': q_data['is_goalkick'] = True
            elif q_type == 'Penalty': q_data['is_penalty'] = True

        event_data: Dict[str, Any] = {
            'id': event.get('id'),
            'eventId': event.get('eventId'),
            'minute': event.get('minute'),
            'second': event.get('second'),
            'period': event.get('period', {}).get('displayName'),
            'team': team_dict.get(event.get('teamId')),
            'playerId': event.get('playerId'),
            'player_name': player_dict.get(str(event.get('playerId'))),
            'x_coord': event.get('x'),
            'y_coord': event.get('y'),
            'end_x': q_data['pass_end_x'], # Behalte die alten Spalten für Kompatibilität
            'end_y': q_data['pass_end_y'],
            'event_type': event.get('type', {}).get('displayName'),
            'outcome': event.get('outcomeType', {}).get('displayName'),
            'isGoal': event.get('isGoal'),
            'isShot': event.get('isShot'),
        }
        # Füge die neuen, strukturierten Qualifier-Daten hinzu
        event_data.update(q_data)

        processed_events.append(event_data)
        
    return processed_events

def parse_formations(data):
    """Extracts formation and lineup data for the match."""
    log.info("Verarbeite Formations-Daten...")
    all_formations: List[Dict[str, Any]] = []
    player_lookup = data.get('playerIdNameDictionary', {})
    teams = [('home', data.get('home', {})), ('away', data.get('away', {}))]
    
    for field, team_data in teams:
        if not team_data:
            continue
        for form in team_data.get('formations', []):
            form_data: Dict[str, Any] = {
                'team_name': team_data.get('name'),
                'field': field,
                'formationName': form.get('formationName'),
                'captainPlayerId': form.get('captainPlayerId'),
                'captainPlayerName': player_lookup.get(str(form.get('captainPlayerId'))),
                'period': form.get('period'),
                'startMinuteExpanded': form.get('startMinuteExpanded'),
                'endMinuteExpanded': form.get('endMinuteExpanded'),
                'playerIds': form.get('playerIds', []),
                'jerseyNumbers': form.get('jerseyNumbers', []),
                'formationSlots': form.get('formationSlots', []),
            }
            all_formations.append(form_data)
            
    return all_formations


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "item"


def _accept_cookies_if_present(page) -> None:
    try:
        agree_button = page.locator('button:has-text("AGREE"), button:has-text("ACCEPT ALL"), button:has-text("ZUSTIMMEN"), button:has-text("ALLE AKZEPTIEREN")')
        agree_button.wait_for(timeout=5000)
        if agree_button.count() > 0:
            log.info("Cookie-Banner gefunden, klicke auf 'Zustimmen'...")
            agree_button.first.click()
    except TimeoutError:
        return


def _normalize_whoscored_url(url_or_path: str) -> str:
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return url_or_path
    if not url_or_path.startswith("/"):
        url_or_path = "/" + url_or_path
    return WHO_SCORED_BASE_URL + url_or_path


def _extract_require_args_object(script_text: str) -> Optional[str]:
    """Extracts the JS object assigned to require.config.params["args"].

    Uses a balanced-brace scan (regex is too fragile for nested objects).
    Returns the substring including the outer braces, or None if not found.
    """
    marker = 'require.config.params["args"]'
    idx = script_text.find(marker)
    if idx == -1:
        return None

    # Find '=' after marker
    eq = script_text.find("=", idx)
    if eq == -1:
        return None

    # Find first '{' after '='
    start = script_text.find("{", eq)
    if start == -1:
        return None

    brace_depth = 0
    in_single = False
    in_double = False
    escape = False

    for pos in range(start, len(script_text)):
        ch = script_text[pos]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if in_single:
            if ch == "'":
                in_single = False
            continue

        if in_double:
            if ch == '"':
                in_double = False
            continue

        if ch == "'":
            in_single = True
            continue

        if ch == '"':
            in_double = True
            continue

        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0:
                return script_text[start : pos + 1]

    return None


def scrape_team_match_urls(fixtures_url: str, *, opts: ScrapeOptions) -> List[str]:
    """Collects match URLs from a WhoScored team fixtures page."""
    match_urls: List[str] = []
    seen: Set[str] = set()

    with sync_playwright() as p:
        log.info("Browser wird gestartet (Fixtures)...")
        browser = p.chromium.launch(headless=opts.headless)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = context.new_page()

        log.info(f"Fixtures werden abgerufen: {fixtures_url}")
        page.goto(fixtures_url, wait_until='domcontentloaded')
        _accept_cookies_if_present(page)

        # Best effort: click "Show more" to load more fixtures
        for _ in range(max(0, opts.show_more_clicks)):
            locator = page.locator('a:has-text("Show more"), button:has-text("Show more"), a:has-text("Mehr"), button:has-text("Mehr")')
            try:
                if locator.count() == 0:
                    break
                # If multiple, click the first visible
                clicked = False
                for i in range(locator.count()):
                    el = locator.nth(i)
                    if el.is_visible():
                        el.click()
                        clicked = True
                        break
                if not clicked:
                    break
                page.wait_for_timeout(800)
            except Exception:
                break

        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, 'html.parser')
    for a in soup.find_all('a', href=True):
        href = a.get('href')
        if not href:
            continue
        # Typical match links look like /matches/<id>/live/<slug>
        if "/matches/" not in href:
            continue
        if "/live/" not in href and "/preview/" not in href and "/matchreport/" not in href:
            continue

        # Competition filter (default: Bundesliga)
        if opts.competition_slug and opts.competition_slug not in href:
            continue

        # Season filter (default: 2025-2026)
        if opts.season_slug and opts.season_slug not in href:
            continue

        url = _normalize_whoscored_url(href)
        if url in seen:
            continue
        seen.add(url)
        match_urls.append(url)

    # Prefer live URLs where possible
    match_urls.sort(key=lambda u: ("/live/" not in u, u))
    if opts.per_team_limit is not None:
        match_urls = match_urls[: max(0, int(opts.per_team_limit))]
    log.info(f"Gefundene Match-Links: {len(match_urls)}")
    return match_urls

def scrape_whoscored_match(url):
    """
    Scrapes all relevant match data from a WhoScored match URL.
    Returns a structured dict with raw + parsed data.
    """
    try:
        with sync_playwright() as p:
            log.info("Browser wird gestartet...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
            )
            page = context.new_page()

            # Ensure we don't hang indefinitely on navigation/waits.
            page.set_default_timeout(45000)
            page.set_default_navigation_timeout(45000)

            log.info(f"Daten werden von folgender URL abgerufen: {url}")

            last_exc: Optional[Exception] = None
            for attempt in range(1, 3):
                try:
                    page.goto(url, wait_until='domcontentloaded', timeout=45000)
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    log.warning(f"Navigation fehlgeschlagen (attempt {attempt}/2): {e}")
                    try:
                        page.wait_for_timeout(800)
                    except Exception:
                        pass
            if last_exc is not None:
                raise last_exc

            log.info("Seite geladen, prüfe auf Cookie-Banner...")
            _accept_cookies_if_present(page)

            log.info("Warte auf das Laden der Spieldaten...")
            page.wait_for_selector('#layout-wrapper', state='attached', timeout=45000)
            page.wait_for_timeout(1500) # Extra wait for JS hydration

            # Many pages populate matchCentreData asynchronously; wait best-effort.
            try:
                page.wait_for_function(
                    """() => {
                        try {
                            return (typeof require !== 'undefined'
                                && require.config
                                && require.config.params
                                && require.config.params["args"]
                                && require.config.params["args"].matchCentreData);
                        } catch (e) {
                            return false;
                        }
                    }""",
                    timeout=45000,
                )
            except Exception:
                pass

            # Preferred: pull the already-materialized args object from the page context.
            log.info("Extrahiere require.config.params['args'] via page.evaluate()...")
            data_args = None
            try:
                data_args = page.evaluate(
                    """() => {
                        try {
                            return (typeof require !== 'undefined' && require.config && require.config.params)
                                ? require.config.params["args"]
                                : null;
                        } catch (e) {
                            return null;
                        }
                    }"""
                )
            except Exception:
                data_args = None

            # Fallback: parse from HTML if evaluate() didn't work.
            if not data_args:
                log.warning("page.evaluate() lieferte keine args-Daten. Fallback: HTML-Extraktion...")
                html_content = page.content()
            else:
                html_content = None

            browser.close()

        if not data_args:
            if not html_content:
                log.error("Fehler: Keine Seite/HTML im Fallback verfügbar.")
                return None

            soup = BeautifulSoup(html_content, 'html.parser')
            scripts = soup.find_all('script')

            data_script = None
            for script in scripts:
                if script.string and 'require.config.params["args"]' in script.string:
                    data_script = script.string
                    break

            if not data_script:
                log.error("Fehler: Konnte das Kern-Daten-Script-Tag nicht finden.")
                return None

            js_object_str = _extract_require_args_object(data_script)
            if not js_object_str:
                log.error("Fehler: Konnte require.config.params[\"args\"] nicht extrahieren (balanced scan).")
                return None

            # Best-effort conversion; may fail on non-JSON JS literals.
            json_str = re.sub(r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', js_object_str)

            try:
                data_args = json.loads(json_str)
            except json.JSONDecodeError as e:
                log.error(f"Fehler beim Parsen von JSON (Fallback): {e}")
                log.error(f"Fehlerhaftes JSON (Ausschnitt): {json_str[:500]}")
                return None

        data = data_args.get('matchCentreData') if isinstance(data_args, dict) else None

        if not data:
            # Not all pages expose matchCentreData (or it may require more time / different endpoint).
            # Still persist what we have for debugging / later enrichment.
            match_id = None
            if isinstance(data_args, dict):
                match_id = data_args.get("matchId")
            log.warning("'matchCentreData' nicht gefunden; speichere Partial-JSON mit raw args.")
            return {
                "status": {
                    "ok": False,
                    "reason": "matchCentreData_missing",
                },
                "source": {
                    "provider": "whoscored",
                    "url": url,
                    "scrapedAt": _now_iso(),
                },
                "raw": {
                    "args": data_args,
                    "matchId": match_id,
                },
                "parsed": {
                    "game_info": None,
                    "team_stats": [],
                    "player_stats": [],
                    "events": [],
                    "formations": [],
                },
            }

        log.info("JSON-Daten erfolgreich extrahiert und geparst.")
        
        # --- DEBUG: Gesamtes Objekt speichern (optional) ---
        # data_dir = 'Data'
        # if not os.path.exists(data_dir):
        #     os.makedirs(data_dir)
        # with open(os.path.join(data_dir, 'debug_full_data.json'), 'w', encoding='utf-8') as f:
        #     json.dump(data, f, indent=4, ensure_ascii=False)
        # log.info("Debug-JSON-Datei gespeichert.")
        # --- ENDE DEBUG ---

        parsed = {
            "game_info": parse_game_info(data),
            "team_stats": parse_team_stats(data),
            "player_stats": parse_player_stats(data),
            "events": parse_events(data),
            "formations": parse_formations(data),
        }

        results: Dict[str, Any] = {
            "status": {
                "ok": True,
            },
            "source": {
                "provider": "whoscored",
                "url": url,
                "scrapedAt": _now_iso(),
            },
            "raw": {
                "matchCentreData": data,
            },
            "parsed": parsed,
        }
        
        log.info("Alle Datenblöcke erfolgreich verarbeitet.")
        return results

    except Exception as e:
        log.error(f"Ein schwerwiegender Fehler ist aufgetreten: {e}", exc_info=True)
        return None


def _default_output_dir() -> str:
    return os.path.join("Data", "whoscored")


def _safe_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def scrape_teams_to_json(team_fixtures: Dict[str, str], *, out_dir: str, opts: ScrapeOptions) -> None:
    index: Dict[str, Any] = {
        "provider": "whoscored",
        "scrapedAt": _now_iso(),
        "teams": {},
    }

    for team_key, fixtures_url in team_fixtures.items():
        log.info(f"\n=== Team: {team_key} ===")
        match_urls = scrape_team_match_urls(fixtures_url, opts=opts)

        team_out_dir = os.path.join(out_dir, team_key)
        os.makedirs(team_out_dir, exist_ok=True)

        saved: List[str] = []
        failed: List[str] = []

        for i, match_url in enumerate(match_urls, start=1):
            # Determine output file name
            slug = match_url.rstrip("/").split("/")[-1]
            if not slug or slug.isdigit():
                slug = _slugify(match_url.rstrip("/").split("/")[-2])
            out_path = os.path.join(team_out_dir, f"{slug}.json")

            if (not opts.force) and os.path.exists(out_path):
                log.info(f"[{i}/{len(match_urls)}] Skip (exists): {slug}")
                saved.append(out_path)
                continue

            log.info(f"[{i}/{len(match_urls)}] Scrape: {match_url}")
            data = scrape_whoscored_match(match_url)
            if not data:
                failed.append(match_url)
                continue

            _safe_write_json(out_path, data)
            saved.append(out_path)

            if opts.delay_seconds > 0:
                time.sleep(opts.delay_seconds)

        index["teams"][team_key] = {
            "fixturesUrl": fixtures_url,
            "matchUrls": match_urls,
            "savedFiles": [os.path.relpath(p, start=out_dir).replace("\\", "/") for p in saved],
            "failedUrls": failed,
        }

    _safe_write_json(os.path.join(out_dir, "index.json"), index)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape WhoScored fixtures + matches to JSON")
    parser.add_argument("--out", default=_default_output_dir(), help="Output directory (default: Data/whoscored)")
    parser.add_argument("--headful", action="store_true", help="Run browser non-headless (debug)")
    parser.add_argument("--show-more-clicks", type=int, default=20, help="How often to click 'Show more'")
    parser.add_argument("--per-team-limit", type=int, default=0, help="Limit matches per team (0 = no limit)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between matches in seconds")
    parser.add_argument("--force", action="store_true", help="Overwrite existing JSON files")
    parser.add_argument(
        "--competition-slug",
        default="germany-bundesliga",
        help="Only scrape matches whose URL contains this competition slug (default: germany-bundesliga)",
    )
    parser.add_argument(
        "--season-slug",
        default="2025-2026",
        help="Only scrape matches whose URL contains this season slug (default: 2025-2026)",
    )
    parser.add_argument(
        "--teams",
        nargs="*",
        default=list(DEFAULT_TEAM_FIXTURES.keys()),
        help=f"Which teams to scrape (keys): {', '.join(DEFAULT_TEAM_FIXTURES.keys())}",
    )
    parser.add_argument(
        "--team-url",
        action="append",
        default=[],
        help="Custom team fixtures mapping as key=url (can be provided multiple times)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()

    per_team_limit = None if args.per_team_limit == 0 else max(1, int(args.per_team_limit))
    opts = ScrapeOptions(
        headless=not args.headful,
        show_more_clicks=max(0, int(args.show_more_clicks)),
        per_team_limit=per_team_limit,
        delay_seconds=max(0.0, float(args.delay)),
        force=bool(args.force),
        competition_slug=str(args.competition_slug or "").strip(),
        season_slug=str(args.season_slug or "").strip(),
    )

    team_fixtures: Dict[str, str] = {}

    # Predefined teams
    for key in args.teams:
        if key in DEFAULT_TEAM_FIXTURES:
            team_fixtures[key] = DEFAULT_TEAM_FIXTURES[key]
        else:
            log.warning(f"Unbekannter Team-Key '{key}' (übersprungen).")

    # Custom mappings
    for pair in args.team_url:
        if "=" not in pair:
            log.warning(f"Ungültiges --team-url Format (erwartet key=url): {pair}")
            continue
        key, url = pair.split("=", 1)
        key = _slugify(key)
        team_fixtures[key] = url.strip()

    if not team_fixtures:
        log.error("Keine Teams zum Scrapen ausgewählt. Nutze z.B. --teams hamburger-sv")
        raise SystemExit(2)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    scrape_teams_to_json(team_fixtures, out_dir=out_dir, opts=opts)
    log.info(f"\nFertig. Index: {os.path.join(out_dir, 'index.json')}")

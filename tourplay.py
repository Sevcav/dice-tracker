"""
tourplay.py
-----------
Fetch the league's match fixtures (teams + coaches) so the dice tracker can
pre-fill player names from the same source as the "Blood Bowl Game Sheets"
site (https://sevcav.github.io/BB-Printed-Game-Sheet/).

TourPlay's API blocks direct calls (HTTP 403 — Cloudflare bot protection),
so we go through the SAME Cloudflare Worker proxy the Game Sheets site uses;
it proxies /api/* -> tourplay.net/api/* with the right origin/CORS. Verified
working server-side 2026-06-17.

Endpoint (mirrors the site's loadFixtures):
  GET {WORKER}/api/tournament/{slug}/phases?page=0&pageSize=75
        &phaseId={phaseId}&type=COACH[&round={n}]
  - no round  -> {rounds:[{roundNumber,...}], currentRound, matches:[...]}
  - &round=N  -> {..., matches:[ {round, group, rosterLocal, rosterVisitor,
                  scoreResume, ...} ]}
Coach name lives at rosterLocal/Visitor.inscription.player.userNameToShow
(falling back to .player.userNameToShow), exactly as the site parses it.
"""

import json
import re
import urllib.parse
import urllib.request

WORKER = "https://bb-gamesheet-app.chapman-thor.workers.dev"
_TIMEOUT = 25


def parse_sheet_url(url: str) -> dict | None:
    """Pull slug / categoryId / phaseId from a Game Sheets URL (the address
    bar of sevcav.github.io/BB-Printed-Game-Sheet/?...). Returns a dict or
    None if the required params aren't present."""
    try:
        q = urllib.parse.urlparse(url).query
        p = urllib.parse.parse_qs(q)
        slug = (p.get("slug") or p.get("league") or [None])[0]
        phase = (p.get("phaseId") or [None])[0]
        if not slug or not phase:
            return None
        return {
            "slug": slug,
            "categoryId": (p.get("categoryId") or [None])[0],
            "phaseId": phase,
        }
    except Exception:
        return None


def _clean_race(race: str) -> str:
    """'Nurgle_BB2025' / 'OldWorldAlliance_BB2025_Legacy' -> 'Nurgle' /
    'OldWorldAlliance'. Mirrors the site's cleanRace (strip the _BB2025
    suffix and any trailing _Legacy)."""
    if not race:
        return ""
    r = re.sub(r"_BB\d{4}.*$", "", race)
    return r.strip()


def _coach(roster: dict) -> str:
    ins = roster.get("inscription") or {}
    p = ins.get("player") or roster.get("player") or {}
    return p.get("userNameToShow", "") or ""


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _phases_base(slug: str, phase_id: str) -> str:
    return (f"{WORKER}/api/tournament/{urllib.parse.quote(slug)}/phases"
            f"?page=0&pageSize=75&phaseId={urllib.parse.quote(str(phase_id))}"
            f"&type=COACH")


def fetch_matches(slug: str, phase_id: str) -> dict:
    """Return {'rounds': [int...], 'current_round': int, 'matches': [...]}.
    Each match: {round, division, home_team, home_race, home_coach,
    away_team, away_race, away_coach}. Fetches each round (the base call
    only returns the current round's matches), same as the site."""
    base = _phases_base(slug, phase_id)
    meta = _get(base)
    rounds_meta = meta.get("rounds") or []
    current = meta.get("currentRound") or 0
    round_nums = [r.get("roundNumber") for r in rounds_meta
                  if r.get("roundNumber")]

    matches: list[dict] = []
    for rn in round_nums:
        try:
            d = _get(f"{base}&round={rn}")
        except Exception:
            continue
        for m in (d.get("matches") or []):
            local = m.get("rosterLocal") or {}
            visitor = m.get("rosterVisitor") or {}
            home_team = local.get("teamName") or ""
            away_team = visitor.get("teamName") or ""
            if not home_team or not away_team:
                continue
            matches.append({
                "round": m.get("round", rn),
                "division": (m.get("group") or {}).get("name", "").upper(),
                "home_team": home_team,
                "home_race": _clean_race(local.get("teamRace", "")),
                "home_coach": _coach(local),
                "away_team": away_team,
                "away_race": _clean_race(visitor.get("teamRace", "")),
                "away_coach": _coach(visitor),
            })
    return {"rounds": round_nums, "current_round": current,
            "matches": matches}

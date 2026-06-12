"""
webapp.py
---------
Phone web UI for the Blood Bowl Dice Tracker (Flask).

Two halves:

1. LIVE CONTROL (only when attached to a running dice_tracker.py):
   - pick the dice type for the next roll (Block / D6 / D16)
   - set the active rolling player (until the physical buttons exist)
   - live status: tracker state, last roll, roll count

2. GAME REVIEW (works any time, including standalone on the PC):
   - list of games
   - per-game roll log with confidences, reject/edit flags
   - THE END-OF-GAME RECORD: per-player, per-dice-type face tallies
   - post-game corrections: edit a roll's faces (flagged 'edited'),
     delete a roll
   - CSV export per game

Run standalone (review only):       python webapp.py [--port 5000]
Attached mode is started automatically by dice_tracker.py.

Phone access: http://<pi-or-pc-ip>:5000/  (same WiFi / hotspot).
"""

import csv
import io
import json
import math
import socket
import threading
import time

from flask import (Flask, Response, redirect, render_template_string,
                   request, url_for)

import db
from dice_types import TYPE_FACES as FACES

DICE_TYPES = ["auto", "block", "d6", "d16"]


class WebControl:
    """Thread-safe mailbox between the Flask thread and the tracker loop.

    The web side writes requests (requested_type / requested_player);
    the tracker loop applies them and clears them, and refreshes `status`
    every frame.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.requested_type: str | None = None
        self.requested_player: str | None = None
        self.status: dict = {}

    def request_type(self, dice_type: str):
        with self._lock:
            self.requested_type = dice_type

    def request_player(self, player: str):
        with self._lock:
            self.requested_player = player

    def take_requests(self) -> tuple[str | None, str | None]:
        with self._lock:
            t, p = self.requested_type, self.requested_player
            self.requested_type = None
            self.requested_player = None
            return t, p

    def update_status(self, **kwargs):
        with self._lock:
            self.status = dict(kwargs, updated_at=time.time())

    def get_status(self) -> dict:
        with self._lock:
            return dict(self.status)


# ── Templates (single-file app: inline, mobile-first) ──────────────────────
_BASE = """
<!doctype html><html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BB Dice Tracker</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #111;
         color: #eee; }
  header { background: #1d3557; padding: 10px 14px; display: flex;
           gap: 14px; align-items: baseline; }
  header a { color: #a8dadc; text-decoration: none; font-weight: 600; }
  main { padding: 14px; max-width: 720px; margin: 0 auto; }
  h1 { font-size: 1.2rem; } h2 { font-size: 1.05rem; margin-top: 1.4em; }
  .btnrow { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0; }
  button, .btn { font-size: 1.1rem; padding: 14px 20px; border-radius: 10px;
           border: 2px solid #457b9d; background: #1d3557; color: #fff;
           text-decoration: none; display: inline-block; }
  button.active { background: #2a9d8f; border-color: #2a9d8f; }
  button.p1 { border-color: #e63946; } button.p1.active { background:#e63946; }
  button.p2 { border-color: #f4a261; } button.p2.active { background:#f4a261;
              color:#111; }
  table { border-collapse: collapse; width: 100%; font-size: 0.92rem; }
  th, td { border-bottom: 1px solid #333; padding: 6px 8px; text-align: left;}
  .rej { color: #e63946; } .edited { color: #f4a261; }
  .tally { display: inline-block; background: #222; border: 1px solid #444;
           border-radius: 8px; padding: 8px 12px; margin: 4px; }
  .tally b { font-size: 1.2rem; }
  .pbanner { padding: 8px 14px; border-radius: 8px; font-weight: 800;
           text-transform: uppercase; letter-spacing: 0.5px;
           margin: 18px 0 8px; }
  .pbanner.p1 { background: #ffd60a; color: #111; }
  .pbanner.p2 { background: #48cae4; color: #111; }
  .facerow { display: flex; flex-wrap: wrap; gap: 8px 16px; margin: 10px 0;
           align-items: flex-start; }
  .face { display: inline-flex; flex-direction: column;
           align-items: center; min-width: 44px; }
  .face b { font-size: 1.05rem; margin-top: 3px; }
  .facelabel { font-size: 0.66rem; color: #9a9a9a; margin-top: 1px;
           white-space: nowrap; }
  .face.dim { opacity: 0.4; }
  .facedie { width: 34px; height: 34px; border-radius: 7px;
           display: inline-flex; align-items: center; justify-content: center;
           font-size: 19px; color: #111;
           box-shadow: 0 1px 3px rgba(0,0,0,0.6); }
  .facedie.bg-cream { background: #ece1bd; }
  .facedie.bg-black { background: #1b1b1b; }
  .d16num { font-weight: 800; font-size: 15px; }
  .rollcard { background: #1a1a1a; border: 1px solid #333;
           border-radius: 10px; padding: 10px 12px; margin: 8px 0; }
  .rollcard .head { font-size: 0.85rem; color: #aaa; }
  .rollcard .faces { font-size: 1.15rem; font-weight: 600; margin: 4px 0; }
  .rollcard details { margin-top: 6px; }
  .rollcard summary { color: #a8dadc; cursor: pointer; font-size: 0.9rem; }
  .tablewrap { overflow-x: auto; }
  .muted { color: #888; font-size: 0.85rem; }
  input[type=text], select { font-size: 1rem; padding: 8px;
           border-radius: 6px; border: 1px solid #555; background: #222;
           color: #eee; margin: 2px 0; }
  form.inline { display: inline; }
  .statusbox { background: #1a1a1a; border: 1px solid #333;
           border-radius: 10px; padding: 10px 14px; margin: 10px 0; }
</style></head><body>
<header>
  <a href="{{ url_for('live') }}">Live</a>
  <a href="{{ url_for('games') }}">Games</a>
</header>
<main>{{ body|safe }}</main>
{{ script|safe }}
</body></html>
"""


def _page(body: str, script: str = "") -> str:
    return render_template_string(_BASE, body=body, script=script)


# ── End-of-game record rendering (styled after the BB3 dice report, with
#    face icons drawn to match the user's ACTUAL 2025 starter-box dice:
#    cream block dice with spiked-ring symbols, black d6 with white pips,
#    cream numbered d16. Reference photos in "Dice Images/". ─────────────────
_D6_ORDER    = ["1pip", "2pip", "3pip", "4pip", "5pip", "6BB"]
_BLOCK_ORDER = ["push", "both_down", "stumble", "pow", "player_down"]
_INK   = "#151515"
_CREAM = "#f1e8cf"
_PIPS = {
    1: [(18, 18)],
    2: [(10, 10), (26, 26)],
    3: [(10, 10), (18, 18), (26, 26)],
    4: [(10, 10), (26, 10), (10, 26), (26, 26)],
    5: [(10, 10), (26, 10), (18, 18), (10, 26), (26, 26)],
}


def _spiked_ring(filled: bool, cx=18.0, cy=18.0, r=12.0,
                 spikes=10, spike_len=3.2) -> str:
    """The chaos-spike ring common to every block die face."""
    parts = []
    for i in range(spikes):
        a = 2 * math.pi * i / spikes
        tx, ty = cx + (r + spike_len) * math.cos(a), \
                 cy + (r + spike_len) * math.sin(a)
        b1x, b1y = cx + r * math.cos(a - 0.14), cy + r * math.sin(a - 0.14)
        b2x, b2y = cx + r * math.cos(a + 0.14), cy + r * math.sin(a + 0.14)
        parts.append(f'<polygon points="{b1x:.1f},{b1y:.1f} {tx:.1f},{ty:.1f}'
                     f' {b2x:.1f},{b2y:.1f}" fill="{_INK}"/>')
    if filled:
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{_INK}"/>')
    else:
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r - 1.3}" fill="none" '
                     f'stroke="{_INK}" stroke-width="2.6"/>')
    return "".join(parts)


def _burst(cx, cy, r_out, r_in, n, fill, rot=0.0) -> str:
    """Comic-style explosion starburst."""
    pts = []
    for i in range(2 * n):
        rr = r_out if i % 2 == 0 else r_in
        a = math.pi * i / n + rot
        pts.append(f"{cx + rr * math.cos(a):.1f},{cy + rr * math.sin(a):.1f}")
    return f'<polygon points="{" ".join(pts)}" fill="{fill}"/>'


def _skull(cx, cy, s, outline=False) -> str:
    st = f' stroke="{_INK}" stroke-width="{1.2 * s:.1f}"' if outline else ""
    return (
        f'<ellipse cx="{cx}" cy="{cy - 1.2 * s:.1f}" rx="{6.2 * s:.1f}" '
        f'ry="{5.8 * s:.1f}" fill="{_CREAM}"{st}/>'
        f'<rect x="{cx - 3.4 * s:.1f}" y="{cy + 3.0 * s:.1f}" '
        f'width="{6.8 * s:.1f}" height="{3.6 * s:.1f}" rx="{1.4 * s:.1f}" '
        f'fill="{_CREAM}"{st}/>'
        f'<circle cx="{cx - 2.4 * s:.1f}" cy="{cy - 1.4 * s:.1f}" '
        f'r="{1.7 * s:.1f}" fill="{_INK}"/>'
        f'<circle cx="{cx + 2.4 * s:.1f}" cy="{cy - 1.4 * s:.1f}" '
        f'r="{1.7 * s:.1f}" fill="{_INK}"/>'
        f'<polygon points="{cx:.1f},{cy + 0.8 * s:.1f} '
        f'{cx - 1.0 * s:.1f},{cy + 2.6 * s:.1f} '
        f'{cx + 1.0 * s:.1f},{cy + 2.6 * s:.1f}" fill="{_INK}"/>'
    )


def _block_face_svg(face: str) -> str:
    if face == "pow":
        body = (_spiked_ring(True)
                + _burst(18, 18, 9.6, 4.0, 11, _CREAM)
                + _burst(18, 18, 5.4, 2.2, 11, _INK)
                + _burst(18, 18, 3.2, 1.4, 11, _CREAM))
    elif face == "push":
        body = (_spiked_ring(True)
                + f'<polygon points="24.9,10.8 15.6,13.0 22.6,20.0" '
                  f'fill="{_CREAM}"/>'
                + f'<polygon points="10.9,22.7 13.3,25.1 20.4,18.0 '
                  f'18.0,15.6" fill="{_CREAM}"/>')
    elif face == "both_down":
        body = (_spiked_ring(True)
                + _burst(23.2, 13.2, 5.6, 2.4, 9, _CREAM)
                + _skull(14.6, 19.8, 0.78))
    elif face == "player_down":
        body = _spiked_ring(False) + _skull(18, 17.8, 1.05, outline=True)
    else:  # stumble — exclamation mark with burst wings
        body = (_spiked_ring(True)
                + _burst(11.3, 16.5, 3.8, 1.7, 8, _CREAM)
                + _burst(24.7, 16.5, 3.8, 1.7, 8, _CREAM)
                + f'<polygon points="16.6,9.5 19.4,9.5 18.6,19.5 17.4,19.5" '
                  f'fill="{_CREAM}"/>'
                + f'<circle cx="18" cy="23.6" r="1.9" fill="{_CREAM}"/>')
    return f'<svg viewBox="0 0 36 36" width="28" height="28">{body}</svg>'


def _d6_face_svg(face: str) -> str:
    """User's BB d6: black die, white pips; the 6 is the BB logo."""
    if face == "6BB":
        inner = ('<text x="18" y="23" font-size="13" font-weight="800" '
                 'font-style="italic" text-anchor="middle" fill="#fff" '
                 'font-family="sans-serif">BB</text>')
    else:
        n = int(face[0])
        inner = "".join(f'<circle cx="{x}" cy="{y}" r="4" fill="#fff"/>'
                        for x, y in _PIPS[n])
    return f'<svg viewBox="0 0 36 36" width="26" height="26">{inner}</svg>'


_FACE_LABEL = {"push": "Push", "both_down": "Both Down",
               "stumble": "Stumble", "pow": "POW!",
               "player_down": "Player Down",
               "1pip": "1", "2pip": "2", "3pip": "3",
               "4pip": "4", "5pip": "5", "6BB": "6 (BB)"}


def _face_chip(inner: str, count: int, title: str,
               bg: str = "bg-cream") -> str:
    dim = ' dim' if count == 0 else ''
    label = _FACE_LABEL.get(title, "")
    label_html = (f'<span class="facelabel">{label}</span>' if label else "")
    return (f'<span class="face{dim}" title="{title}">'
            f'<span class="facedie {bg}">{inner}</span>'
            f'<b>{count}</b>{label_html}</span>')


def _player_record_html(tally: dict) -> str:
    """One player's dice record: d6 row, block row, d16 row (if rolled)."""
    d6 = tally.get("d6", {})
    block = tally.get("block", {})
    d16 = tally.get("d16", {})

    rows = '<div class="facerow">' + "".join(
        _face_chip(_d6_face_svg(face), d6.get(face, 0), face, "bg-black")
        for face in _D6_ORDER) + "</div>"
    rows += '<div class="facerow">' + "".join(
        _face_chip(_block_face_svg(face), block.get(face, 0), face)
        for face in _BLOCK_ORDER) + "</div>"
    rows += '<div class="facerow">' + "".join(
        _face_chip(f'<span class="d16num">{n}</span>',
                   d16.get(f"D16_{n}", 0), f"D16_{n}")
        for n in range(1, 17)) + "</div>"
    # anything outside the known vocabularies (manual entries etc.)
    known = set(_D6_ORDER) | set(_BLOCK_ORDER) | {f"D16_{n}"
                                                  for n in range(1, 17)}
    extras = [(t, f, c) for t, faces in tally.items()
              for f, c in faces.items() if f not in known]
    if extras:
        rows += ('<p class="muted">other: '
                 + ", ".join(f"{t}/{f}: {c}" for t, f, c in extras)
                 + "</p>")
    return rows


def _fix_fields(roll: dict) -> str:
    """Edit controls for a roll: one dropdown per die (+ remove option and
    an add-die dropdown) when the face vocabulary is known; free-text
    fallback otherwise (future d8/d3 manual entries)."""
    vocab = FACES.get(roll["dice_type"])
    if vocab is None:
        results = ", ".join(roll["results"])
        return f'<input type="text" name="results" value="{results}" size="16">'
    html = ""
    for face in roll["results"]:
        # keep a non-vocab value (old free-text edit) selectable
        opts = ([face] if face not in vocab else []) + vocab
        options = "".join(
            f'<option value="{v}"{" selected" if v == face else ""}>{v}'
            f'</option>' for v in opts)
        options += '<option value="">(remove)</option>'
        html += f'<select name="face">{options}</select> '
    add_options = '<option value="" selected>(add die)</option>' + "".join(
        f'<option value="{v}">{v}</option>' for v in vocab)
    html += f'<select name="face">{add_options}</select> '
    return html


# ── App factory ─────────────────────────────────────────────────────────────
def create_app(control: WebControl | None = None) -> Flask:
    app = Flask(__name__)
    db.init_db()

    # ── Live control ────────────────────────────────────────────────────
    @app.route("/")
    def live():
        if control is None:
            return redirect(url_for("games"))
        body = """
<h1>Live <span id="statechip" class="state-chip">connecting...</span></h1>
<div id="daymode"></div>

<h2>Current read</h2>
<div id="dice" class="dicerow"><span class="muted">no dice in tray</span>
</div>
<p class="muted" id="hint"></p>

<h2>Dice type</h2>
<div class="btnrow" id="dicebtns">
  <button onclick="setType('auto')" id="t-auto">Auto</button>
  <button onclick="setType('block')" id="t-block">Block</button>
  <button onclick="setType('d6')" id="t-d6">D6</button>
  <button onclick="setType('d16')" id="t-d16">D16</button>
</div>
<h2>Rolling player</h2>
<div class="btnrow">
  <button class="p1" onclick="setPlayer('P1')" id="p-P1">Player 1</button>
  <button class="p2" onclick="setPlayer('P2')" id="p-P2">Player 2</button>
</div>

<h2>Recent rolls <span class="muted" id="rollcount"></span></h2>
<ul id="recent" class="recent"><li class="muted">none yet</li></ul>
<p class="muted">Confirm / reject / undo happen on the rig. Orange ? =
the model is unsure - nudge that die to re-read it before confirming.
D8 / D3: enter manually in the game log after the game.</p>
"""
        script = """
<style>
  .dicerow { display: flex; gap: 10px; flex-wrap: wrap; min-height: 84px;
             align-items: center; }
  .die { border-radius: 12px; padding: 12px 16px; text-align: center;
         font-weight: 700; font-size: 1.15rem; min-width: 84px; }
  .die small { display: block; font-weight: 400; font-size: 0.8rem;
         opacity: 0.85; }
  .die.sure { background: #2a9d8f; color: #fff; }
  .die.unsure { background: #f4a261; color: #111;
         outline: 3px solid #e76f51; }
  .die.settling { background: #333; color: #aaa; }
  .state-chip { font-size: 0.95rem; padding: 4px 12px; border-radius: 999px;
         background: #333; vertical-align: middle; }
  .state-chip.settled { background: #2a9d8f; }
  .state-chip.confirmed { background: #6d597a; }
  ul.recent { list-style: none; padding: 0; }
  ul.recent li { border-bottom: 1px solid #333; padding: 7px 2px; }
</style>
<script>
async function setType(t) {
  await fetch('/api/dice_type', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({type:t})});
  poll();
}
async function setPlayer(p) {
  await fetch('/api/player', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({player:p})});
  poll();
}
const HINTS = {
  watching: 'Roll the dice...',
  settled: 'Read locked - check it, then confirm on the rig',
  confirmed: 'Logged. Pick the dice up to arm the next roll.'
};
async function poll() {
  try {
    const r = await fetch('/api/status'); const s = await r.json();
    const age = s.updated_at ? (Date.now()/1000 - s.updated_at) : 1e9;
    const chip = document.getElementById('statechip');
    if (age > 5) { chip.textContent = 'tracker offline';
                   chip.className = 'state-chip'; return; }
    const typeLabel = s.dice_type === 'auto'
      ? 'auto' + (s.detected_type ? ' [' + s.detected_type + ']' : '')
      : s.dice_type;
    chip.textContent = s.state + ' - ' + typeLabel;
    chip.className = 'state-chip ' + s.state;
    document.getElementById('hint').textContent = HINTS[s.state] || '';
    document.getElementById('daymode').innerHTML = s.day_mode
      ? '<div class="statusbox" style="border-color:#e63946;color:#e63946">'
        + '<b>DAY MODE - check the light shield!</b></div>' : '';

    const dice = s.dice || [];
    document.getElementById('dice').innerHTML = dice.length
      ? dice.map(d => `<div class="die ${
            d.stable ? (d.uncertain ? 'unsure' : 'sure') : 'settling'}">
            ${d.label}${d.uncertain || !d.stable ? '?' : ''}
            <small>${d.conf}%${d.uncertain ? ' - nudge?' : ''}</small>
          </div>`).join('')
      : '<span class="muted">no dice in tray</span>';

    document.getElementById('rollcount').textContent =
      '(' + s.rolls + ' logged)';
    const rec = s.recent || [];
    document.getElementById('recent').innerHTML = rec.length
      ? rec.map(t => `<li>${t}</li>`).join('')
      : '<li class="muted">none yet</li>';

    for (const t of ['auto','block','d6','d16'])
      document.getElementById('t-'+t).classList.toggle('active',
        s.dice_type === t);
    for (const p of ['P1','P2'])
      document.getElementById('p-'+p).classList.toggle('active',
        s.player === p);
    if (s.p1_name) document.getElementById('p-P1').textContent = s.p1_name;
    if (s.p2_name) document.getElementById('p-P2').textContent = s.p2_name;
  } catch (e) {}
}
setInterval(poll, 700); poll();
</script>
"""
        return _page(body, script)

    @app.post("/api/dice_type")
    def api_dice_type():
        t = (request.json or {}).get("type")
        if control is not None and t in DICE_TYPES:
            control.request_type(t)
            return {"ok": True}
        return {"ok": False}, 400

    @app.post("/api/player")
    def api_player():
        p = (request.json or {}).get("player")
        if control is not None and p in ("P1", "P2"):
            control.request_player(p)
            return {"ok": True}
        return {"ok": False}, 400

    @app.get("/api/status")
    def api_status():
        return control.get_status() if control is not None else {}

    # ── Game review ─────────────────────────────────────────────────────
    @app.route("/games")
    def games():
        rows = db.list_games()
        items = ""
        for g in rows:
            started = time.strftime("%Y-%m-%d %H:%M",
                                    time.localtime(g["started_at"]))
            open_tag = "" if g["ended_at"] else " (in progress)"
            items += (f'<tr><td><a class="btn" style="padding:6px 12px" '
                      f'href="/games/{g["id"]}">Game {g["id"]}</a></td>'
                      f'<td>{started}{open_tag}</td>'
                      f'<td>{g["player1_name"]} vs {g["player2_name"]}</td>'
                      f'<td>{g["roll_count"]} rolls</td></tr>')
        body = (f"<h1>Games</h1><div class='tablewrap'><table>"
                f"<tr><th></th><th>Started</th>"
                f"<th>Players</th><th>Rolls</th></tr>{items}</table></div>"
                if items else "<h1>Games</h1><p>No games recorded yet.</p>")
        return _page(body)

    @app.route("/games/<int:game_id>")
    def game_detail(game_id):
        g = db.get_game(game_id)
        if g is None:
            return _page("<p>Game not found.</p>"), 404
        rolls = db.get_rolls(game_id)
        tallies = db.face_tallies(game_id)
        names = {"P1": g["player1_name"], "P2": g["player2_name"]}

        # The end-of-game record: per-player face tallies, styled after
        # the BB3 dice report (banner + icon-per-face counts).
        tally_html = ""
        for player, css in (("P1", "p1"), ("P2", "p2")):
            tally_html += (f'<div class="pbanner {css}">'
                           f'{names[player]}</div>')
            if not tallies[player]:
                tally_html += '<p class="muted">No confirmed rolls.</p>'
            else:
                tally_html += _player_record_html(tallies[player])

        roll_cards = ""
        for r in reversed(rolls):       # newest first on the phone
            ts = time.strftime("%H:%M:%S", time.localtime(r["timestamp"]))
            flags = ""
            if r["rejected"]:
                flags += ' <span class="rej">rejected</span>'
            if r["edited"]:
                flags += ' <span class="edited">edited</span>'
            confs = " ".join(f"{int(c*100)}%" for c in r["confidences"])
            results = ", ".join(r["results"])
            pname = names.get(r["player"], r["player"])
            roll_cards += f"""
<div class="rollcard">
  <div class="head">#{r['roll_no']} &middot; {ts} &middot; {pname}
       &middot; {r['dice_type']}{flags}</div>
  <div class="faces">{results}</div>
  <div class="muted">{confs}</div>
  <details><summary>fix</summary>
    <form class="inline" method="post"
          action="/games/{game_id}/rolls/{r['id']}/edit">
      {_fix_fields(r)}
      <button style="padding:6px 10px;font-size:0.85rem">save</button>
    </form>
    <form class="inline" method="post"
          action="/games/{game_id}/rolls/{r['id']}/delete"
          onsubmit="return confirm('Delete roll #{r['roll_no']}?')">
      <button style="padding:6px 10px;font-size:0.85rem">delete</button>
    </form>
  </details>
</div>"""

        started = time.strftime("%Y-%m-%d %H:%M",
                                time.localtime(g["started_at"]))
        body = f"""
<h1>Game {game_id} — {names['P1']} vs {names['P2']}</h1>
<p class="muted">Started {started}.
<a class="btn" style="padding:6px 12px"
   href="/games/{game_id}/export.csv">Export CSV</a></p>
<h1>Dice record</h1>
{tally_html}
<h1>Roll log <span class="muted">(newest first)</span></h1>
{roll_cards or '<p class="muted">No rolls yet.</p>'}
<p class="muted">Edits replace the faces for a roll (comma-separated) and
are flagged. Rejected rolls are excluded from the record.</p>
"""
        script = f"""
<script>
// Auto-refresh when a new roll lands, but never while the user is typing
// in an edit field.
const RENDERED_COUNT = {len(rolls)};
async function checkNew() {{
  const el = document.activeElement;
  if (el && ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) return;
  // also hold the refresh while any fix panel is open
  if (document.querySelector('details[open]')) return;
  try {{
    const r = await fetch('/api/games/{game_id}/roll_count');
    const j = await r.json();
    if (j.count !== RENDERED_COUNT) location.reload();
  }} catch (e) {{}}
}}
setInterval(checkNew, 2500);
</script>
"""
        return _page(body, script)

    @app.get("/api/games/<int:game_id>/roll_count")
    def api_roll_count(game_id):
        return {"count": db.roll_count(game_id)}

    @app.post("/games/<int:game_id>/rolls/<int:roll_id>/edit")
    def roll_edit(game_id, roll_id):
        if "face" in request.form:
            # dropdown form: blanks are removed dice / unused add-die slot
            results = [f for f in request.form.getlist("face") if f]
        else:
            raw = request.form.get("results", "")
            results = [t.strip() for t in raw.replace(",", " ").split()
                       if t.strip()]
        if results:
            db.edit_roll_results(roll_id, results)
        return redirect(url_for("game_detail", game_id=game_id))

    @app.post("/games/<int:game_id>/rolls/<int:roll_id>/delete")
    def roll_delete(game_id, roll_id):
        db.delete_roll(roll_id)
        return redirect(url_for("game_detail", game_id=game_id))

    @app.get("/games/<int:game_id>/export.csv")
    def game_export(game_id):
        rolls = db.get_rolls(game_id)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["roll_no", "timestamp", "player", "dice_type",
                    "results", "confidences", "rejected", "edited"])
        for r in rolls:
            w.writerow([r["roll_no"], r["timestamp"], r["player"],
                        r["dice_type"], "|".join(r["results"]),
                        "|".join(str(c) for c in r["confidences"]),
                        r["rejected"], r["edited"]])
        return Response(
            buf.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition":
                     f"attachment; filename=game_{game_id}.csv"})

    return app


def lan_ip() -> str:
    """Best-guess LAN address of this machine (DHCP moves it around — print
    the real URL rather than a placeholder). Prefers home-LAN ranges and
    skips VPN/CGNAT (100.64-127.*) and link-local addresses, because the
    default route may go through a VPN tunnel the phone can't reach."""
    ips: list[str] = []
    try:
        ips.extend(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))      # no traffic sent; just picks a route
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass

    def usable(ip: str) -> bool:
        if ip.startswith(("127.", "169.254.")):
            return False
        parts = ip.split(".")
        if parts[0] == "100" and 64 <= int(parts[1]) <= 127:
            return False                # CGNAT / VPN (Tailscale, etc.)
        return True

    candidates = [ip for ip in ips if usable(ip)]
    for prefix in ("192.168.", "10.", "172."):
        for ip in candidates:
            if ip.startswith(prefix):
                return ip
    return candidates[0] if candidates else "localhost"


def start_in_thread(control: WebControl, port: int = 5000) -> threading.Thread:
    """Run the web app in a daemon thread (called by dice_tracker.py)."""
    app = create_app(control)

    def _run():
        # threaded=True is required: the live page polls /api/status every
        # ~0.7s and the default single-threaded dev server makes page
        # navigation queue behind held-open poll connections (page "freezes").
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False,
                threaded=True)

    t = threading.Thread(target=_run, daemon=True, name="webapp")
    t.start()
    return t


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Standalone review server")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    print(f"Review server (phone): http://{lan_ip()}:{args.port}/games")
    print(f"Review server (this PC): http://localhost:{args.port}/games")
    create_app(None).run(host="0.0.0.0", port=args.port, threaded=True)

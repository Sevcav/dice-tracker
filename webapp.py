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
import html
import io
import json
import math
import socket
import threading
import time

from flask import (Flask, Response, redirect, render_template_string,
                   request, url_for)

import db
import tourplay
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
        self.requested_action: str | None = None   # reject / undo / confirm
        self.requested_names: tuple | None = None   # (p1_name, p2_name)
        self.status: dict = {}
        # Headless support: the tracker pushes the latest JPEG here so the
        # phone can show a live view (camera alignment + monitoring) without
        # any monitor on the Pi. align_confirmed is set when the operator
        # taps Confirm on the phone alignment screen.
        self.latest_jpeg: bytes | None = None
        self.align_confirmed: bool = False
        # Phone tap-4-corners re-calibration: the phone posts 4 tray
        # corners (in streamed-frame pixel coords); the tracker loop picks
        # them up, rewrites tray_roi.json, and reloads the ROI live.
        self.new_corners: list | None = None
        # True only while the startup alignment loop is streaming the green
        # outline. Once the session goes live the stream carries the live
        # read instead, so /align shows a "session running" notice rather
        # than a confusing broken image. Alignment is a startup-only step.
        self.aligning: bool = False

    def request_type(self, dice_type: str):
        with self._lock:
            self.requested_type = dice_type

    def request_player(self, player: str):
        with self._lock:
            self.requested_player = player

    def request_action(self, action: str):
        with self._lock:
            self.requested_action = action

    def request_names(self, p1: str, p2: str):
        with self._lock:
            self.requested_names = (p1, p2)

    def take_names(self) -> tuple | None:
        with self._lock:
            n = self.requested_names
            self.requested_names = None
            return n

    def take_requests(self) -> tuple[str | None, str | None]:
        with self._lock:
            t, p = self.requested_type, self.requested_player
            self.requested_type = None
            self.requested_player = None
            return t, p

    def take_action(self) -> str | None:
        with self._lock:
            a = self.requested_action
            self.requested_action = None
            return a

    def set_frame(self, jpeg: bytes):
        with self._lock:
            self.latest_jpeg = jpeg

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self.latest_jpeg

    def confirm_alignment(self):
        with self._lock:
            self.align_confirmed = True

    def set_new_corners(self, corners: list, frame_w: int, frame_h: int):
        with self._lock:
            self.new_corners = (corners, frame_w, frame_h)

    def take_new_corners(self):
        with self._lock:
            c = self.new_corners
            self.new_corners = None
            return c

    def take_alignment(self) -> bool:
        with self._lock:
            v = self.align_confirmed
            self.align_confirmed = False
            return v

    def set_aligning(self, on: bool):
        with self._lock:
            self.aligning = on

    def is_aligning(self) -> bool:
        with self._lock:
            return self.aligning

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
  header { background: #1d3557; padding: 8px 10px; display: flex;
           gap: 8px; align-items: stretch;
           position: sticky; top: 0; z-index: 10; }
  header a { color: #a8dadc; text-decoration: none; font-weight: 600;
           flex: 1; text-align: center; padding: 12px 6px;
           border-radius: 8px; background: #16243f; font-size: 1.05rem; }
  header a:active { background: #2a4a7a; }
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
  .rollcard.pinned { border: 2px solid #457b9d; background: #16243f; }
  .rollcard.pinned .faces { font-size: 1.4rem; }
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
  <a href="{{ url_for('align') }}">Align</a>
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
_GAMES_IMPORT_JS = """
<script>
let SHEET_MATCHES = [];
function loadSheet(){
  const url = document.getElementById('sheeturl').value.trim();
  const msg = document.getElementById('sheetmsg');
  msg.textContent = 'Fetching from TourPlay...';
  fetch('/api/gamesheet/matches', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({url})})
    .then(r=>r.json()).then(function(d){
      if(!d.ok){ msg.textContent = d.error || 'Fetch failed.'; return; }
      SHEET_MATCHES = d.matches || [];
      const rs = document.getElementById('roundsel');
      rs.innerHTML = '';
      (d.rounds||[]).forEach(function(rn){
        const o=document.createElement('option'); o.value=rn;
        o.textContent='Round '+rn; if(rn===d.current_round)o.selected=true;
        rs.appendChild(o);
      });
      rs.style.display = 'inline-block';
      document.getElementById('matchsel').style.display = 'inline-block';
      fillMatches();
      msg.textContent = SHEET_MATCHES.length + ' matches loaded. Pick a round + match.';
    }).catch(function(e){ msg.textContent = 'Error: '+e; });
}
function fillMatches(){
  const rn = parseInt(document.getElementById('roundsel').value, 10);
  const ms = document.getElementById('matchsel');
  ms.innerHTML = '<option value="">-- pick match --</option>';
  SHEET_MATCHES.filter(m=>m.round===rn).forEach(function(m, i){
    const idx = SHEET_MATCHES.indexOf(m);
    const o=document.createElement('option'); o.value=idx;
    o.textContent = m.home_coach+' ('+m.home_team+')  vs  '
                  + m.away_coach+' ('+m.away_team+')';
    ms.appendChild(o);
  });
}
function pickMatch(){
  const idx = document.getElementById('matchsel').value;
  if(idx==='') return;
  const m = SHEET_MATCHES[parseInt(idx,10)];
  // Use coach names as the player names (fall back to team if no coach).
  document.getElementById('p1in').value = m.home_coach || m.home_team;
  document.getElementById('p2in').value = m.away_coach || m.away_team;
  document.getElementById('sheetmsg').textContent =
    'Filled: '+document.getElementById('p1in').value+' vs '
    +document.getElementById('p2in').value+' — now tap Save names.';
}
// Auto-load matches on page open if the URL is already filled (league
// default), so you go straight to picking a round + match.
window.addEventListener('DOMContentLoaded', function(){
  const u = document.getElementById('sheeturl');
  if (u && u.value.trim()) loadSheet();
});
</script>
"""


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

    @app.post("/api/action")
    def api_action():
        a = (request.json or {}).get("action")
        if control is not None and a in ("reject", "undo", "confirm"):
            control.request_action(a)
            return {"ok": True}
        return {"ok": False}, 400

    # ── Headless camera view + phone-driven alignment ───────────────────
    # The rig has no monitor (sealed box); the phone is the only screen.
    # The tracker pushes JPEG frames into WebControl; these routes serve
    # them and let the operator confirm camera alignment from the phone.
    @app.get("/stream.mjpg")
    def stream_mjpg():
        if control is None:
            return Response(status=404)

        def gen():
            import time as _t
            boundary = b"--frame"
            while True:
                jpg = control.get_frame()
                if jpg is not None:
                    yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpg + b"\r\n")
                _t.sleep(0.05)   # ~20 fps cap

        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/align_state")
    def api_align_state():
        return {"aligning": bool(control and control.is_aligning())}

    @app.route("/align")
    def align():
        body = """
<h2>Camera alignment</h2>
<div id="notalign" style="display:none">
  <p>A session is already running — alignment is a <b>startup-only</b>
  step, so the live read is showing on the <b>Live</b> tab instead.</p>
  <p class="muted">To re-align (e.g. the camera got bumped), restart the
  rig service, then come back here before confirming:</p>
  <pre style="background:#222;padding:10px;border-radius:6px;overflow:auto">sudo systemctl restart dice-tracker</pre>
</div>
<div id="aligning" style="display:none">
<p>Adjust the camera arm until the tray edges line up with the GREEN
outline, then tap Confirm. (Alignment is always step one — the model only
knows the calibrated tray perspective.)</p>
<p class="muted">Moved the camera? Tap <b>Re-set corners</b>, then tap the
tray's 4 corners on the image in order: <b>top-left, top-right,
bottom-right, bottom-left</b>. Save to recalibrate.</p>
<div style="position:relative;display:inline-block;max-width:640px;width:100%">
  <img id="feed" data-src="/stream.mjpg"
       style="width:100%;display:block;border:2px solid #444;border-radius:8px">
  <canvas id="ov" style="position:absolute;left:0;top:0;width:100%;height:100%;
       pointer-events:none"></canvas>
</div>
<div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
  <button class="big" onclick="confirmAlign()">Confirm alignment</button>
  <button class="big" onclick="startPick()">Re-set corners</button>
  <button class="big" id="saveBtn" onclick="saveCorners()"
          style="display:none">Save corners</button>
  <button class="big" id="undoBtn" onclick="undoCorner()"
          style="display:none">Undo last</button>
</div>
<p id="msg" style="margin-top:10px;color:#6c6"></p>
</div>
"""
        script = """
<script>
var feed=document.getElementById('feed'),ov=document.getElementById('ov'),
    msg=document.getElementById('msg');
var picking=false, pts=[];
var LABELS=['top-left','top-right','bottom-right','bottom-left'];

// Show the alignment UI only during the startup alignment step; otherwise
// show the "session running" notice (alignment is startup-only).
fetch('/api/align_state').then(r=>r.json()).then(function(s){
  document.getElementById('aligning').style.display = s.aligning?'block':'none';
  document.getElementById('notalign').style.display = s.aligning?'none':'block';
  // Only open the MJPEG stream during alignment so a live session has no
  // phantom stream client sitting on this page.
  if(s.aligning){ feed.src = feed.getAttribute('data-src'); fit(); }
});

function fit(){ ov.width=feed.clientWidth; ov.height=feed.clientHeight; draw(); }
window.addEventListener('resize',fit);
feed.addEventListener('load',fit);

function draw(){
  var c=ov.getContext('2d'); c.clearRect(0,0,ov.width,ov.height);
  if(!pts.length) return;
  c.lineWidth=2; c.strokeStyle='#3f6'; c.fillStyle='#3f6';
  c.beginPath();
  for(var i=0;i<pts.length;i++){
    var p=pts[i]; if(i===0)c.moveTo(p.x,p.y); else c.lineTo(p.x,p.y);
  }
  if(pts.length===4) c.closePath();
  c.stroke();
  for(var i=0;i<pts.length;i++){
    var p=pts[i];
    c.beginPath(); c.arc(p.x,p.y,5,0,7); c.fill();
    c.fillStyle='#fff'; c.fillText(String(i+1),p.x+7,p.y-7); c.fillStyle='#3f6';
  }
}
function startPick(){
  picking=true; pts=[];
  ov.style.pointerEvents='auto';
  document.getElementById('saveBtn').style.display='none';
  document.getElementById('undoBtn').style.display='inline-block';
  msg.textContent='Tap corner 1: '+LABELS[0];
  fit();
}
function undoCorner(){
  pts.pop(); draw();
  if(pts.length<4) document.getElementById('saveBtn').style.display='none';
  msg.textContent = pts.length<4
    ? 'Tap corner '+(pts.length+1)+': '+LABELS[pts.length]
    : 'All 4 set — Save corners.';
}
ov.addEventListener('click',function(e){
  if(!picking||pts.length>=4) return;
  var r=ov.getBoundingClientRect();
  pts.push({x:e.clientX-r.left, y:e.clientY-r.top});
  draw();
  if(pts.length<4){ msg.textContent='Tap corner '+(pts.length+1)+': '
        +LABELS[pts.length]; }
  else { msg.textContent='All 4 set — Save corners.';
        document.getElementById('saveBtn').style.display='inline-block'; }
});
function saveCorners(){
  if(pts.length!==4){msg.textContent='Tap all 4 corners first.';return;}
  // Map displayed coords -> the stream's native pixel size.
  var sx=feed.naturalWidth/feed.clientWidth,
      sy=feed.naturalHeight/feed.clientHeight;
  var c=pts.map(function(p){return [Math.round(p.x*sx),Math.round(p.y*sy)];});
  fetch('/api/align_corners',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({corners:c,
          frame_width:feed.naturalWidth, frame_height:feed.naturalHeight})})
    .then(r=>r.json()).then(function(d){
      if(d.ok){ picking=false; ov.style.pointerEvents='none';
        document.getElementById('saveBtn').style.display='none';
        document.getElementById('undoBtn').style.display='none';
        msg.textContent='Saved. Green outline updated — Confirm to start.';
      } else { msg.textContent='Save failed: '+(d.error||'?'); }
    });
}
function confirmAlign(){
  fetch('/api/align_confirm',{method:'POST'})
    .then(r=>r.json()).then(_=>{msg.textContent='Confirmed — starting...';});
}
</script>
"""
        return _page(body, script)

    @app.post("/api/align_confirm")
    def api_align_confirm():
        if control is not None:
            control.confirm_alignment()
            return {"ok": True}
        return {"ok": False}, 400

    @app.post("/api/align_corners")
    def api_align_corners():
        if control is None:
            return {"ok": False, "error": "no control"}, 400
        data = request.get_json(silent=True) or {}
        corners = data.get("corners")
        fw = data.get("frame_width")
        fh = data.get("frame_height")
        if (not isinstance(corners, list) or len(corners) != 4
                or not fw or not fh):
            return {"ok": False, "error": "need 4 corners + frame size"}, 400
        control.set_new_corners(corners, int(fw), int(fh))
        return {"ok": True}

    # ── Game review ─────────────────────────────────────────────────────
    def _active_game_id():
        """The game the live tracker is currently logging to (or None) —
        published into status each frame; protected from deletion."""
        if control is None:
            return None
        return control.get_status().get("active_game_id")

    def _current_names():
        """Names to prefill the form: live session names if the tracker is
        running, else the most recent game's, else defaults."""
        if control is not None:
            st = control.get_status()
            if st.get("p1_name") or st.get("p2_name"):
                return (st.get("p1_name", "Player 1"),
                        st.get("p2_name", "Player 2"))
        rows = db.list_games()
        if rows:
            return rows[0]["player1_name"], rows[0]["player2_name"]
        return "Player 1", "Player 2"

    @app.route("/games")
    def games():
        rows = db.list_games()
        active = _active_game_id()
        cur_p1, cur_p2 = _current_names()
        names_form = (
            '<form method="post" action="/games/names" '
            'style="background:#1a1a1a;border:1px solid #333;'
            'border-radius:10px;padding:12px;margin:10px 0">'
            '<div style="font-weight:600;margin-bottom:8px">Player names</div>'
            '<div style="display:flex;gap:8px;flex-wrap:wrap;'
            'align-items:center">'
            f'<input type="text" id="p1in" name="p1" '
            f'value="{html.escape(cur_p1)}" '
            'placeholder="Player 1" style="flex:1;min-width:120px">'
            f'<input type="text" id="p2in" name="p2" '
            f'value="{html.escape(cur_p2)}" '
            'placeholder="Player 2" style="flex:1;min-width:120px">'
            '<button class="big">Save names</button></div>'
            '<p class="muted" style="margin:8px 0 0">Applies to the next '
            'game; renames the in-progress game too.</p></form>')
        sheet_url = html.escape(_sheet["url"])
        import_block = (
            '<div style="background:#1a1a1a;border:1px solid #333;'
            'border-radius:10px;padding:12px;margin:10px 0">'
            '<div style="font-weight:600;margin-bottom:8px">'
            'Load from Game Sheets</div>'
            f'<input type="text" id="sheeturl" value="{sheet_url}" '
            'placeholder="Paste Game Sheets URL (with slug &amp; phaseId)" '
            'style="width:100%;box-sizing:border-box;margin-bottom:8px">'
            '<div style="display:flex;gap:8px;flex-wrap:wrap;'
            'align-items:center">'
            '<button type="button" class="big" onclick="loadSheet()">'
            'Fetch matches</button>'
            '<select id="roundsel" onchange="fillMatches()" '
            'style="display:none"></select>'
            '<select id="matchsel" onchange="pickMatch()" '
            'style="display:none;flex:1;min-width:160px"></select>'
            '</div>'
            '<p class="muted" id="sheetmsg" style="margin:8px 0 0">'
            'Pick a match to fill the names below, then Save names.</p>'
            '</div>')
        items = ""
        for g in rows:
            started = time.strftime("%Y-%m-%d %H:%M",
                                    time.localtime(g["started_at"]))
            open_tag = "" if g["ended_at"] else " (in progress)"
            is_active = (g["id"] == active)
            if is_active:
                del_cell = '<span class="muted">active</span>'
            else:
                del_cell = (
                    f'<form class="inline" method="post" '
                    f'action="/games/{g["id"]}/delete" '
                    f'onsubmit="return confirm('
                    f"'Delete Game {g['id']} and its rolls?')\">"
                    f'<button style="padding:6px 10px;font-size:0.85rem">'
                    f'delete</button></form>')
            items += (f'<tr><td><a class="btn" style="padding:6px 12px" '
                      f'href="/games/{g["id"]}">Game {g["id"]}</a></td>'
                      f'<td>{started}{open_tag}</td>'
                      f'<td>{g["player1_name"]} vs {g["player2_name"]}</td>'
                      f'<td>{g["roll_count"]} rolls</td>'
                      f'<td>{del_cell}</td></tr>')
        if items:
            clear_btn = (
                '<form method="post" action="/games/clear" '
                'style="margin:14px 0" '
                "onsubmit=\"return confirm('Delete ALL games"
                + (" except the active one" if active else "")
                + "? This cannot be undone.')\">"
                '<button class="big">Clear all games</button></form>')
            body = (f"<h1>Games</h1>{import_block}{names_form}{clear_btn}"
                    f"<div class='tablewrap'><table>"
                    f"<tr><th></th><th>Started</th>"
                    f"<th>Players</th><th>Rolls</th><th></th></tr>"
                    f"{items}</table></div>")
        else:
            body = (f"<h1>Games</h1>{import_block}{names_form}"
                    f"<p class='muted'>No games recorded yet. Names above "
                    f"apply to the next game.</p>")
        return _page(body, _GAMES_IMPORT_JS)

    # Remembered Game Sheets URL so you don't re-paste it each round.
    # Seeded with the league default so the box is pre-filled and the page
    # auto-loads matches without typing.
    _sheet = {"url": tourplay.DEFAULT_SHEET_URL}

    @app.post("/api/gamesheet/matches")
    def api_gamesheet_matches():
        """Fetch league fixtures (teams + coaches) from the Game Sheets
        TourPlay source via the Cloudflare Worker. Body: {url} (a Game
        Sheets URL with slug/phaseId)."""
        data = request.get_json(silent=True) or {}
        url = (data.get("url") or _sheet["url"] or "").strip()
        params = tourplay.parse_sheet_url(url)
        if not params:
            return {"ok": False,
                    "error": "Paste a Game Sheets URL containing slug "
                             "and phaseId."}, 400
        try:
            res = tourplay.fetch_matches(params["slug"], params["phaseId"])
        except Exception as e:
            return {"ok": False,
                    "error": f"Fetch failed (Pi online?): {e}"}, 502
        _sheet["url"] = url
        return {"ok": True, "current_round": res["current_round"],
                "rounds": res["rounds"], "matches": res["matches"]}

    @app.post("/games/names")
    def games_set_names():
        p1 = (request.form.get("p1") or "Player 1").strip()
        p2 = (request.form.get("p2") or "Player 2").strip()
        if control is not None:
            control.request_names(p1, p2)
        # If no tracker is running but a game exists, rename the latest
        # directly so the change isn't silently lost in review-only mode.
        elif (rows := db.list_games()):
            db.set_player_names(rows[0]["id"], p1, p2)
        return redirect(url_for("games"))

    @app.post("/games/<int:game_id>/delete")
    def game_delete(game_id):
        if game_id == _active_game_id():
            return _page("<p>Can't delete the game that's currently being "
                         "recorded. <a href='/games'>Back</a></p>"), 409
        db.delete_game(game_id)
        return redirect(url_for("games"))

    @app.post("/games/clear")
    def games_clear():
        db.delete_all_games(except_id=_active_game_id())
        return redirect(url_for("games"))

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

        def _roll_card(r, *, open_fix=False, pinned=False):
            ts = time.strftime("%H:%M:%S", time.localtime(r["timestamp"]))
            flags = ""
            if r["rejected"]:
                flags += ' <span class="rej">rejected</span>'
            if r["edited"]:
                flags += ' <span class="edited">edited</span>'
            confs = " ".join(f"{int(c*100)}%" for c in r["confidences"])
            results = ", ".join(r["results"])
            pname = names.get(r["player"], r["player"])
            cls = "rollcard pinned" if pinned else "rollcard"
            openattr = " open" if open_fix else ""
            # The pinned card's fix panel is open by default; mark it so the
            # auto-refresh guard ignores it (only MANUALLY opened panels on
            # the log below should hold the refresh).
            detcls = ' class="pinnedfix"' if pinned else ""
            summary = "edit / delete" if pinned else "fix"
            return f"""
<div class="{cls}">
  <div class="head">#{r['roll_no']} &middot; {ts} &middot; {pname}
       &middot; {r['dice_type']}{flags}</div>
  <div class="faces">{results}</div>
  <div class="muted">{confs}</div>
  <details{openattr}{detcls}><summary>{summary}</summary>
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

        # Latest roll pinned at the very top, fix panel open for quick
        # correction (the main reason the user sits on this page). The full
        # log below shows every roll newest-first, including this one.
        latest = rolls[-1] if rolls else None
        latest_html = (_roll_card(latest, open_fix=True, pinned=True)
                       if latest else
                       '<p class="muted">No rolls yet.</p>')
        roll_cards = "".join(_roll_card(r) for r in reversed(rolls))

        started = time.strftime("%Y-%m-%d %H:%M",
                                time.localtime(g["started_at"]))
        body = f"""
<h1>Game {game_id} — {names['P1']} vs {names['P2']}</h1>
<h2>Latest roll</h2>
{latest_html}
<h1>Dice record</h1>
{tally_html}
<h1>Roll log <span class="muted">(newest first)</span></h1>
{roll_cards or '<p class="muted">No rolls yet.</p>'}
<p class="muted">Edits replace the faces for a roll (comma-separated) and
are flagged. Rejected rolls are excluded from the record.</p>
<hr style="border:none;border-top:1px solid #333;margin:18px 0">
<p class="muted">Started {started}.
<a class="btn" style="padding:6px 12px"
   href="/games/{game_id}/export.csv">Export CSV</a></p>
"""
        script = f"""
<script>
// Auto-refresh when a new roll lands, but never while the user is typing
// in an edit field.
const RENDERED_COUNT = {len(rolls)};
async function checkNew() {{
  const el = document.activeElement;
  if (el && ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) return;
  // hold the refresh while a MANUALLY opened fix panel is open, but not
  // for the pinned latest-roll card (its panel is always open).
  if (document.querySelector('details[open]:not(.pinnedfix)')) return;
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

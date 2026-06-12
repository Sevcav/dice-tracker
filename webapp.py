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

        # The end-of-game record: per-player face tallies
        tally_html = ""
        for player in ("P1", "P2"):
            tally_html += f"<h2>{names[player]} ({player})</h2>"
            if not tallies[player]:
                tally_html += '<p class="muted">No confirmed rolls.</p>'
            for dice_type, faces in sorted(tallies[player].items()):
                total = sum(faces.values())
                chips = "".join(
                    f'<span class="tally">{face}<br><b>{n}</b></span>'
                    for face, n in sorted(faces.items(),
                                          key=lambda kv: -kv[1]))
                tally_html += (f"<p>{dice_type} — {total} dice</p>"
                               f"<div>{chips}</div>")

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
    print(f"Review server: http://localhost:{args.port}/games")
    create_app(None).run(host="0.0.0.0", port=args.port, threaded=True)

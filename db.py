"""
db.py
-----
SQLite persistence for the Blood Bowl Dice Tracker.

Implements the schema drafted in DESIGN.md section 5, slightly extended:
a `games` table so every session is grouped, and an `edited` flag on rolls
so post-game corrections made through the web app stay auditable.

The database is a single file (dice_tracker.db) next to this module —
ideal for the Pi. Connections are short-lived (open per call), which is
plenty fast at dice-roll volume and safe across the tracker thread and
the Flask thread.

session.json / session.csv remain as secondary outputs of dice_tracker.py;
the database is the primary record the web app reads.
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "dice_tracker.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    REAL NOT NULL,
    ended_at      REAL,
    player1_name  TEXT NOT NULL DEFAULT 'Player 1',
    player2_name  TEXT NOT NULL DEFAULT 'Player 2',
    notes         TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS rolls (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id        INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    roll_no        INTEGER NOT NULL,
    timestamp      REAL NOT NULL,
    player         TEXT NOT NULL CHECK (player IN ('P1', 'P2')),
    dice_type      TEXT NOT NULL,
    results        TEXT NOT NULL,   -- JSON list, e.g. ["pow","push"]
    confidences    TEXT NOT NULL,   -- JSON list of floats
    rejected       INTEGER NOT NULL DEFAULT 0,
    edited         INTEGER NOT NULL DEFAULT 0,
    raw_image_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_rolls_game ON rolls(game_id);
"""


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db():
    with _connect() as con:
        con.executescript(_SCHEMA)


# ── Game lifecycle ──────────────────────────────────────────────────────────
def create_game(player1_name="Player 1", player2_name="Player 2") -> int:
    with _connect() as con:
        cur = con.execute(
            "INSERT INTO games (started_at, player1_name, player2_name) "
            "VALUES (?, ?, ?)",
            (time.time(), player1_name, player2_name))
        return cur.lastrowid


def end_game(game_id: int):
    with _connect() as con:
        con.execute("UPDATE games SET ended_at = ? WHERE id = ?",
                    (time.time(), game_id))


def set_player_names(game_id: int, p1: str, p2: str):
    with _connect() as con:
        con.execute(
            "UPDATE games SET player1_name = ?, player2_name = ? "
            "WHERE id = ?", (p1, p2, game_id))


def get_game(game_id: int) -> dict | None:
    with _connect() as con:
        row = con.execute("SELECT * FROM games WHERE id = ?",
                          (game_id,)).fetchone()
        return dict(row) if row else None


def list_games() -> list[dict]:
    with _connect() as con:
        rows = con.execute(
            "SELECT g.*, COUNT(r.id) AS roll_count "
            "FROM games g LEFT JOIN rolls r ON r.game_id = g.id "
            "GROUP BY g.id ORDER BY g.started_at DESC").fetchall()
        return [dict(r) for r in rows]


# ── Rolls ───────────────────────────────────────────────────────────────────
def add_roll(game_id: int, roll_no: int, player: str, dice_type: str,
             results: list[str], confidences: list[float],
             rejected: bool = False,
             raw_image_path: str | None = None) -> int:
    with _connect() as con:
        cur = con.execute(
            "INSERT INTO rolls (game_id, roll_no, timestamp, player, "
            "dice_type, results, confidences, rejected, raw_image_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (game_id, roll_no, time.time(), player, dice_type,
             json.dumps(results), json.dumps(confidences),
             int(rejected), raw_image_path))
        return cur.lastrowid


def delete_last_roll(game_id: int) -> dict | None:
    """Undo support: remove and return the most recent roll of a game."""
    with _connect() as con:
        row = con.execute(
            "SELECT * FROM rolls WHERE game_id = ? "
            "ORDER BY id DESC LIMIT 1", (game_id,)).fetchone()
        if row is None:
            return None
        con.execute("DELETE FROM rolls WHERE id = ?", (row["id"],))
        return dict(row)


def delete_roll(roll_id: int):
    with _connect() as con:
        con.execute("DELETE FROM rolls WHERE id = ?", (roll_id,))


def edit_roll_results(roll_id: int, results: list[str]):
    """Post-game correction path: replace results, flag as edited."""
    with _connect() as con:
        con.execute(
            "UPDATE rolls SET results = ?, edited = 1 WHERE id = ?",
            (json.dumps(results), roll_id))


def roll_count(game_id: int) -> int:
    with _connect() as con:
        return con.execute("SELECT COUNT(*) FROM rolls WHERE game_id = ?",
                           (game_id,)).fetchone()[0]


def get_rolls(game_id: int) -> list[dict]:
    with _connect() as con:
        rows = con.execute(
            "SELECT * FROM rolls WHERE game_id = ? ORDER BY id",
            (game_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["results"] = json.loads(d["results"])
        d["confidences"] = json.loads(d["confidences"])
        out.append(d)
    return out


# ── The end-of-game record: per-player face tallies ─────────────────────────
def face_tallies(game_id: int) -> dict:
    """{player: {dice_type: {face: count}}} over confirmed (non-rejected)
    rolls — the per-face record the project exists to produce."""
    tallies: dict = {"P1": {}, "P2": {}}
    for roll in get_rolls(game_id):
        if roll["rejected"]:
            continue
        by_type = tallies[roll["player"]].setdefault(roll["dice_type"], {})
        for face in roll["results"]:
            by_type[face] = by_type.get(face, 0) + 1
    return tallies

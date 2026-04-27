"""
game/state.py
-------------
Blood Bowl dice tracker — game state for a 16-turn, 2-player game.

Tracks per player per turn:
  - d6 Blood Bowl die  : numeric totals (1-6) + BB logo count
  - Block die          : symbol counts (both_down, pow, push, player_down, stumble)
  - d8                 : numeric totals (1-8)
  - d16                : numeric totals (1-16)

Usage:
    from game.state import GameState
    gs = GameState("Team Chaos", "Team Order")
    gs.set_active_player(0)        # player index 0 or 1
    gs.record_roll(results)        # list of DieResult objects
    gs.confirm_turn()              # lock in current turn, advance turn counter
    gs.undo_last_roll()            # revert last recorded roll
    gs.save("session.json")
    gs.load("session.json")
"""

from __future__ import annotations
import json
import os
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any

TOTAL_TURNS    = 16
BLOCK_SYMBOLS  = ["both_down", "pow", "push", "player_down", "stumble"]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class PlayerTurnData:
    """All dice results for one player's single turn."""
    turn:          int = 0
    player_index:  int = 0

    # d6 Blood Bowl die
    d6_rolls:    list[int]   = field(default_factory=list)   # 1-5 pip values
    d6_bb_count: int         = 0                              # number of BB logo (6) results

    # Block die
    block_rolls: list[str]   = field(default_factory=list)   # e.g. ["push","skull"]
    block_counts: dict[str, int] = field(
        default_factory=lambda: {s: 0 for s in BLOCK_SYMBOLS})

    # d8
    d8_rolls:    list[int]   = field(default_factory=list)

    # d16
    d16_rolls:   list[int]   = field(default_factory=list)


@dataclass
class PlayerSummary:
    """Running totals across all turns for one player."""
    name: str

    # d6 — track each pip face count individually (1-5 pips + BB logo face)
    # Mirrors the Blood Bowl video game stat screen exactly
    d6_pip_counts: dict[str, int] = field(
        default_factory=lambda: {'1':0,'2':0,'3':0,'4':0,'5':0,'6':0})
    # '6' key = BB logo face count

    @property
    def d6_roll_count(self) -> int:
        return sum(self.d6_pip_counts.values())

    @property
    def d6_total(self) -> int:
        """Numeric sum of all d6 pip rolls (treating BB logo as 6)."""
        return sum(int(k) * v for k, v in self.d6_pip_counts.items())

    @property
    def d6_bb_total(self) -> int:
        return self.d6_pip_counts.get('6', 0)

    # Block die
    block_counts:   dict[str, int] = field(
        default_factory=lambda: {s: 0 for s in BLOCK_SYMBOLS})
    block_roll_count: int = 0

    # d8 — track each face value individually
    d8_face_counts: dict[str, int] = field(
        default_factory=lambda: {str(i):0 for i in range(1,9)})

    @property
    def d8_roll_count(self) -> int:
        return sum(self.d8_face_counts.values())

    @property
    def d8_total(self) -> int:
        return sum(int(k) * v for k, v in self.d8_face_counts.items())

    # d16 — track each face value individually
    d16_face_counts: dict[str, int] = field(
        default_factory=lambda: {str(i):0 for i in range(1,17)})

    @property
    def d16_roll_count(self) -> int:
        return sum(self.d16_face_counts.values())

    @property
    def d16_total(self) -> int:
        return sum(int(k) * v for k, v in self.d16_face_counts.items())

    def d6_average(self) -> float:
        return self.d6_total / self.d6_roll_count if self.d6_roll_count else 0.0

    def d8_average(self) -> float:
        return self.d8_total / self.d8_roll_count if self.d8_roll_count else 0.0

    def d16_average(self) -> float:
        return self.d16_total / self.d16_roll_count if self.d16_roll_count else 0.0


# ── Main game state ────────────────────────────────────────────────────────────

class GameState:
    def __init__(self, player1_name: str = "Player 1", player2_name: str = "Player 2"):
        self.players       = [player1_name, player2_name]
        self.current_turn  = 1          # 1-16
        self.active_player = 0          # 0 or 1
        self.game_over     = False
        self.started_at    = datetime.now().isoformat()

        self.summaries: list[PlayerSummary] = [
            PlayerSummary(name=player1_name),
            PlayerSummary(name=player2_name),
        ]

        # Full turn-by-turn history (list of PlayerTurnData)
        self.history: list[PlayerTurnData] = []

        # Current pending turn data (not yet confirmed)
        self._pending: PlayerTurnData | None = None

        # Undo stack: list of snapshots
        self._undo_stack: list[dict] = []

    # ── Player/turn management ─────────────────────────────────────────────────

    def set_active_player(self, player_index: int):
        """Set which player (0 or 1) is currently rolling.
        Resets pending so the new player starts a fresh roll slate."""
        assert player_index in (0, 1)
        self.active_player = player_index
        # Always start fresh pending for the newly active player
        self._pending = PlayerTurnData(
            turn=self.current_turn,
            player_index=player_index,
            block_counts={s: 0 for s in BLOCK_SYMBOLS}
        )

    def advance_turn(self):
        """
        Increment the turn counter only.
        Does NOT switch the active player — players switch via set_active_player().
        """
        if self.current_turn >= TOTAL_TURNS:
            self.game_over = True
            return
        self.current_turn += 1
        self._pending = PlayerTurnData(
            turn=self.current_turn,
            player_index=self.active_player,
            block_counts={s: 0 for s in BLOCK_SYMBOLS}
        )

    # ── Recording rolls ────────────────────────────────────────────────────────

    def record_roll(self, die_results: list) -> None:
        """
        Record a set of die results for the current active player's turn.
        Call this once per detected-and-settled roll set.

        Parameters
        ----------
        die_results : list of DieResult objects (from reading/read_die.py)
        """
        if self._pending is None:
            self.set_active_player(self.active_player)

        # Save undo snapshot before modifying
        self._undo_stack.append(self._snapshot())

        for result in die_results:
            if result.raw_value is None:
                continue   # skip unreadable dice

            if result.die_type == "d6_bb":
                if result.raw_value == 6:
                    self._pending.d6_bb_count += 1
                elif isinstance(result.raw_value, int):
                    self._pending.d6_rolls.append(result.raw_value)

            elif result.die_type == "block":
                sym = result.raw_value
                self._pending.block_rolls.append(sym)
                if sym in self._pending.block_counts:
                    self._pending.block_counts[sym] += 1

            elif result.die_type == "d8":
                if isinstance(result.raw_value, int):
                    self._pending.d8_rolls.append(result.raw_value)

            elif result.die_type == "d16":
                if isinstance(result.raw_value, int):
                    self._pending.d16_rolls.append(result.raw_value)

    def confirm_turn(self) -> PlayerTurnData:
        """
        Lock in the current pending roll, update summaries, return the data.

        Does NOT advance the turn or switch the active player.
        Players stay active until they press 1/2 to switch, or N to advance turn.
        """
        if self._pending is None:
            raise RuntimeError("No pending turn to confirm.")

        turn_data = deepcopy(self._pending)
        self.history.append(turn_data)
        self._update_summary(turn_data)
        # Reset pending so the next roll starts fresh for the same player/turn
        self._pending = PlayerTurnData(
            turn=self.current_turn,
            player_index=self.active_player,
            block_counts={s: 0 for s in BLOCK_SYMBOLS}
        )
        return turn_data

    def _update_summary(self, turn: PlayerTurnData):
        s = self.summaries[turn.player_index]

        # d6 — count each pip value individually
        for v in turn.d6_rolls:
            key = str(v)
            if key in s.d6_pip_counts:
                s.d6_pip_counts[key] += 1
        # BB logo face counts as "6"
        s.d6_pip_counts['6'] += turn.d6_bb_count

        # block
        for sym, count in turn.block_counts.items():
            s.block_counts[sym] += count
        s.block_roll_count += len(turn.block_rolls)

        # d8 — count each face value
        for v in turn.d8_rolls:
            key = str(v)
            if key in s.d8_face_counts:
                s.d8_face_counts[key] += 1

        # d16 — count each face value
        for v in turn.d16_rolls:
            key = str(v)
            if key in s.d16_face_counts:
                s.d16_face_counts[key] += 1

    # ── Undo ───────────────────────────────────────────────────────────────────

    def undo_last_roll(self) -> bool:
        """
        Revert the last record_roll() call.
        Returns True if successful, False if nothing to undo.
        """
        if not self._undo_stack:
            return False
        snapshot = self._undo_stack.pop()
        self._restore_snapshot(snapshot)
        return True

    def _snapshot(self) -> dict:
        return {
            "pending":       deepcopy(self._pending),
            "summaries":     deepcopy(self.summaries),
            "history":       deepcopy(self.history),
            "current_turn":  self.current_turn,
            "active_player": self.active_player,
        }

    def _restore_snapshot(self, snap: dict):
        self._pending       = snap["pending"]
        self.summaries      = snap["summaries"]
        self.history        = snap["history"]
        self.current_turn   = snap["current_turn"]
        self.active_player  = snap["active_player"]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save full game state to JSON."""
        data = {
            "players":       self.players,
            "current_turn":  self.current_turn,
            "active_player": self.active_player,
            "game_over":     self.game_over,
            "started_at":    self.started_at,
            "summaries":     [asdict(s) for s in self.summaries],
            "history":       [asdict(h) for h in self.history],
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[GameState] Saved to {path}")

    def load(self, path: str):
        """Load game state from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.players       = data["players"]
        self.current_turn  = data["current_turn"]
        self.active_player = data["active_player"]
        self.game_over     = data["game_over"]
        self.started_at    = data.get("started_at", "")
        self.summaries     = [PlayerSummary(**s) for s in data["summaries"]]
        self.history       = [PlayerTurnData(**h) for h in data["history"]]
        print(f"[GameState] Loaded from {path}")

    def export_csv(self, path: str):
        """Export full roll-by-roll history to CSV."""
        import csv
        rows = []
        for td in self.history:
            player = self.players[td.player_index]
            # d6 rolls
            for v in td.d6_rolls:
                rows.append([player, td.turn, "d6", v])
            for _ in range(td.d6_bb_count):
                rows.append([player, td.turn, "d6", "BB Logo (6)"])
            # block
            for sym in td.block_rolls:
                rows.append([player, td.turn, "Block", sym])
            # d8
            for v in td.d8_rolls:
                rows.append([player, td.turn, "d8", v])
            # d16
            for v in td.d16_rolls:
                rows.append([player, td.turn, "d16", v])

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Player", "Turn", "Die Type", "Result"])
            writer.writerows(rows)
        print(f"[GameState] CSV exported to {path}")

    def save_all(self, base_path: str):
        """Save JSON + CSV + summary report in one call."""
        base = os.path.splitext(base_path)[0]
        self.save(base + ".json")
        self.export_csv(base + ".csv")
        self.export_summary_report(base + "_report.txt")
        print(f"[GameState] All files saved with base: {base}")

    # ── Display helpers ────────────────────────────────────────────────────────

    def status_lines(self, player_index: int) -> list[str]:
        """Return scoreboard lines for one player (for the UI overlay)."""
        s      = self.summaries[player_index]
        name   = self.players[player_index]
        marker = " <<" if player_index == self.active_player else ""
        pc     = s.d6_pip_counts
        bc     = s.block_counts
        lines  = [
            f"{name}{marker}",
            # d6: show each face count like the BB video game
            f"  d6:  1={pc['1']} 2={pc['2']} 3={pc['3']} 4={pc['4']} 5={pc['5']} BB={pc['6']}",
            f"  d8 : {s.d8_total:>4}  (rolls: {s.d8_roll_count})",
            f"  d16: {s.d16_total:>4}  (rolls: {s.d16_roll_count})",
            f"  Push={bc['push']} POW={bc['pow']} BD={bc['both_down']}",
            f"  PD={bc['player_down']} Stbl={bc['stumble']}",
        ]
        return lines

    def export_summary_report(self, path: str):
        """
        Export a styled summary report matching the Blood Bowl video game
        stat screen — per-face counts for d6, per-symbol counts for block dice.
        """
        lines = []
        lines.append("=" * 50)
        lines.append("  BLOOD BOWL DICE TRACKER — GAME SUMMARY")
        lines.append(f"  {self.started_at[:10]}   Turns played: {self.current_turn - 1}/16")
        lines.append("=" * 50)

        for pi, s in enumerate(self.summaries):
            lines.append(f"\n{'─'*50}")
            lines.append(f"  {s.name.upper()}")
            lines.append(f"{'─'*50}")

            # d6 pip counts — each face on its own line with icon-style label
            lines.append("\n  D6 ROLLS:")
            for i, face in enumerate(['1','2','3','4','5','6']):
                count = s.d6_pip_counts[face]
                label = f"{face}pip" if i < 5 else "BB  "
                bar   = '|' * count
                lines.append(f"    [{label}] : {count:>3}  {bar}")

            # Block dice
            lines.append("\n  BLOCK DICE:")
            block_icons = {
                'push':        '->  Push        ',
                'pow':         '**  POW!        ',
                'both_down':   '<>  Both Down   ',
                'player_down': 'XX  Player Down ',
                'stumble':     '!   Stumble     ',
            }
            for sym, icon in block_icons.items():
                count = s.block_counts[sym]
                bar   = '|' * count
                lines.append(f"    {icon} : {count:>3}  {bar}")

            # d8 face counts
            if s.d8_roll_count > 0:
                lines.append(f"\n  D8 ROLLS  (total: {s.d8_total}  avg: {s.d8_average():.1f}):")
                row = "    " + "  ".join(
                    f"{f}:{s.d8_face_counts[f]}" for f in [str(i) for i in range(1,9)]
                    if s.d8_face_counts[f] > 0
                )
                lines.append(row)

            # d16 face counts
            if s.d16_roll_count > 0:
                lines.append(f"\n  D16 ROLLS  (total: {s.d16_total}  avg: {s.d16_average():.1f}):")
                row = "    " + "  ".join(
                    f"{f}:{s.d16_face_counts[f]}" for f in [str(i) for i in range(1,17)]
                    if s.d16_face_counts[f] > 0
                )
                lines.append(row)

        lines.append(f"\n{'='*50}\n")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[GameState] Summary report saved to {path}")

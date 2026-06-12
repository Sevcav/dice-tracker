"""
ui/overlay.py
-------------
Draws the live camera feed + scoreboard overlay window.

Layout:
┌──────────────────────────────┬───────────────────────────┐
│                              │  TURN  7 / 16             │
│   LIVE CAMERA FEED           │  ─────────────────────    │
│                              │  Player 1  ◄              │
│  [d6:3] [d6:BB] [Block:Push] │  d6:  42  BB logo: x3    │
│  [d8:7] [d16:14]             │  d8:  18  d16: 14         │
│                              │  Block: Push=5 POW=2 ...  │
│                              │  ─────────────────────    │
│                              │  Player 2                  │
│                              │  d6:  31  BB logo: x1     │
│                              │  ...                      │
│                              │  ─────────────────────    │
│                              │  [SPACE] Confirm Roll     │
│                              │  [BKSP]  Undo             │
│                              │  [1/2]   Switch Player    │
│                              │  [N]     Next Turn        │
│                              │  [S]     Save Game        │
│                              │  [Q]     Quit             │
└──────────────────────────────┴───────────────────────────┘

Colours:
  - Active player name: bright yellow
  - Confirmed results:  white
  - Pending (not yet confirmed) results: cyan
  - Die bounding boxes: orange (black d6), teal (cream dice)
  - Settled indicator: green flashing border
"""

import cv2
import numpy as np
from game.state import GameState

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
C_WHITE   = (255, 255, 255)
C_CYAN    = (255, 220,  50)
C_YELLOW  = ( 30, 220, 255)
C_GREEN   = ( 60, 220,  60)
C_RED     = ( 60,  60, 220)
C_ORANGE  = ( 30, 140, 255)
C_TEAL    = (180, 200,  30)
C_DARK    = ( 20,  20,  20)
C_PANEL   = ( 30,  30,  30)
C_DIVIDER = ( 80,  80,  80)

PANEL_WIDTH   = 380   # scoreboard panel width (pixels)
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SM       = 0.45
FONT_MD       = 0.55
FONT_LG       = 0.70
LINE_H_SM     = 22
LINE_H_MD     = 26


def _text(img, text, pos, scale, colour, thickness=1):
    cv2.putText(img, text, pos, FONT, scale, colour, thickness, cv2.LINE_AA)


def draw_scoreboard_panel(
    game: GameState,
    pending_labels: list[str] | None,
    settled: bool,
    height: int,
) -> np.ndarray:
    """
    Build the right-hand scoreboard panel as a numpy array.
    Returns a BGR image of shape (height, PANEL_WIDTH, 3).
    """
    panel = np.full((height, PANEL_WIDTH, 3), C_PANEL, dtype=np.uint8)
    x_pad = 12
    y = 18

    # ── Turn header ───────────────────────────────────────────────────────────
    turn_text = f"TURN  {game.current_turn} / 16"
    _text(panel, turn_text, (x_pad, y), FONT_LG, C_YELLOW, 2)
    y += 36

    # Settled indicator
    if settled:
        _text(panel, "  *** DICE SETTLED ***", (x_pad, y), FONT_SM, C_GREEN, 1)
    else:
        _text(panel, "  Rolling...", (x_pad, y), FONT_SM, C_DIVIDER, 1)
    y += LINE_H_SM + 4

    # ── Pending roll labels ───────────────────────────────────────────────────
    if pending_labels:
        _text(panel, "Pending:", (x_pad, y), FONT_SM, C_CYAN, 1)
        y += LINE_H_SM
        line = "  " + "  ".join(pending_labels)
        # Word-wrap if too long
        words = pending_labels
        row = "  "
        for w in words:
            if len(row) + len(w) + 2 > 42:
                _text(panel, row, (x_pad, y), FONT_SM, C_CYAN, 1)
                y += LINE_H_SM
                row = "  " + w
            else:
                row += w + "  "
        if row.strip():
            _text(panel, row, (x_pad, y), FONT_SM, C_CYAN, 1)
        y += LINE_H_SM + 4

    cv2.line(panel, (x_pad, y), (PANEL_WIDTH - x_pad, y), C_DIVIDER, 1)
    y += 8

    # ── Per-player scores ─────────────────────────────────────────────────────
    for pi in range(2):
        s      = game.summaries[pi]
        active = (pi == game.active_player)
        name_colour = C_YELLOW if active else C_WHITE
        marker = " [ROLLING]" if active else ""

        _text(panel, f"{s.name}{marker}", (x_pad, y), FONT_MD, name_colour, 1)
        y += LINE_H_MD

        # d6
        avg6 = f"avg {s.d6_average():.1f}" if s.d6_roll_count else "no rolls"
        _text(panel, f"  d6  : {s.d6_total:>5}  BB-logo x{s.d6_bb_total}  ({avg6})",
              (x_pad, y), FONT_SM, C_WHITE, 1)
        y += LINE_H_SM

        # d8
        avg8 = f"avg {s.d8_average():.1f}" if s.d8_roll_count else "no rolls"
        _text(panel, f"  d8  : {s.d8_total:>5}  rolls={s.d8_roll_count}  ({avg8})",
              (x_pad, y), FONT_SM, C_WHITE, 1)
        y += LINE_H_SM

        # d16
        avg16 = f"avg {s.d16_average():.1f}" if s.d16_roll_count else "no rolls"
        _text(panel, f"  d16 : {s.d16_total:>5}  rolls={s.d16_roll_count}  ({avg16})",
              (x_pad, y), FONT_SM, C_WHITE, 1)
        y += LINE_H_SM

        # Block die
        bc = s.block_counts
        _text(panel, f"  Block ({s.block_roll_count} rolls):", (x_pad, y), FONT_SM, C_WHITE, 1)
        y += LINE_H_SM
        block_str = (f"    Push={bc['push']}  POW={bc['pow']}  "
                     f"BD={bc['both_down']}")
        _text(panel, block_str, (x_pad, y), FONT_SM, C_WHITE, 1)
        y += LINE_H_SM
        block_str2 = f"    PD={bc['player_down']}  Stbl={bc['stumble']}"
        _text(panel, block_str2, (x_pad, y), FONT_SM, C_WHITE, 1)
        y += LINE_H_SM + 4

        if pi == 0:
            cv2.line(panel, (x_pad, y), (PANEL_WIDTH - x_pad, y), C_DIVIDER, 1)
            y += 8

    # ── Keybind help ──────────────────────────────────────────────────────────
    if y < height - 120:
        y = height - 118
    cv2.line(panel, (x_pad, y), (PANEL_WIDTH - x_pad, y), C_DIVIDER, 1)
    y += 8
    help_lines = [
        "[SPACE]  Confirm roll",
        "[BKSP]   Undo last roll",
        "[M]  Manual entry",
        "[1] / [2]  Switch active player",
        "[N]  Next turn manually",
        "[S]  Save game",
        "[P]  Screenshot",
        "[Q]  Quit",
    ]
    for hl in help_lines:
        _text(panel, hl, (x_pad, y), FONT_SM, C_DIVIDER, 1)
        y += LINE_H_SM

    return panel


def draw_frame(
    camera_frame: np.ndarray,
    game: GameState,
    detections,                      # list[DiceDetection]
    die_labels: list[str] | None,    # per-detection label strings
    settled: bool,
    flash_green: bool = False,
    tray_roi: tuple | None = None,   # (x, y, w, h) red tray bounding box
) -> np.ndarray:
    """
    Compose the full display frame:
      left  = camera feed with bounding boxes
      right = scoreboard panel

    Returns the combined BGR image ready for cv2.imshow().
    """
    feed_h, feed_w = camera_frame.shape[:2]
    target_h = max(feed_h, 600)

    # ── Draw bounding boxes on camera feed ────────────────────────────────────
    feed = camera_frame.copy()

    # Draw tray ROI box (blue dashed-style — thin double rect)
    if tray_roi is not None:
        tx, ty, tw, th = tray_roi
        cv2.rectangle(feed, (tx, ty), (tx+tw, ty+th), (200, 80, 0), 1)
        cv2.putText(feed, "TRAY", (tx + 4, ty + 16),
                    FONT, 0.40, (200, 80, 0), 1, cv2.LINE_AA)

    if settled and flash_green:
        # Green border to indicate dice are settled
        cv2.rectangle(feed, (4, 4), (feed_w - 4, feed_h - 4), C_GREEN, 6)

    for i, det in enumerate(detections):
        x, y, w, h = det.bbox
        colour = C_ORANGE if det.colour_hint == "black" else C_TEAL
        cv2.rectangle(feed, (x, y), (x + w, y + h), colour, 2)

        # Label strip above box: "Die N: result confidence%"
        result_text = die_labels[i] if die_labels and i < len(die_labels) else "?"
        label = f"#{i+1}: {result_text}"
        cv2.rectangle(feed, (x, y - 22), (x + len(label) * 9 + 6, y), colour, -1)
        cv2.putText(feed, label, (x + 3, y - 6), FONT, FONT_SM, C_DARK, 1, cv2.LINE_AA)

        # Large die number drawn inside the box (top-left corner)
        # so it's visible even when the label strip is cut off at frame edge
        num_str = str(i + 1)
        cv2.putText(feed, num_str, (x + 4, y + 20),
                    FONT, 0.65, colour, 2, cv2.LINE_AA)

    # ── Resize feed to target height ──────────────────────────────────────────
    scale  = target_h / feed_h
    new_w  = int(feed_w * scale)
    feed   = cv2.resize(feed, (new_w, target_h))

    # ── Build scoreboard panel ────────────────────────────────────────────────
    pending_labels = die_labels if die_labels else []
    panel  = draw_scoreboard_panel(game, pending_labels, settled, target_h)

    # ── Combine side by side ─────────────────────────────────────────────────
    combined = np.concatenate([feed, panel], axis=1)
    return combined

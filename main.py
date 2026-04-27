"""
main.py
-------
Blood Bowl Dice Tracker — main entry point.

Usage:
    python main.py                         # USB webcam (index 0)
    python main.py --camera 1              # second USB camera
    python main.py --camera http://...     # IP / Arduino stream (Phase 2)
    python main.py --player1 "Chaos" --player2 "Order"
    python main.py --load session.json     # resume saved game

Keyboard controls (window must be focused):
    SPACE   Confirm current roll and record it
    BKSP    Undo last roll
    M       Manual entry — type die results when detection fails
    1       Set Player 1 as active
    2       Set Player 2 as active
    N       Manually advance to next turn
    S       Save game to session.json
    Q / ESC Quit (auto-saves)
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

# ── Add project root to path ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from detection.camera            import Camera
from detection.detect_dice       import detect_dice, find_tray_roi, DiceStabilityTracker
from detection.classify_die_type import classify_die_type
from reading.read_die            import DieReader, DieResult
from game.state                  import GameState, BLOCK_SYMBOLS
from ui.overlay                  import draw_frame

SAVE_PATH = os.path.join(BASE, "session.json")

# Minimum confidence to accept a reading — below this shows '?' on screen
CONFIDENCE_THRESHOLD = 0.70

# Flash green border for this many frames after dice settle
FLASH_FRAMES = 30


# ── Manual entry ───────────────────────────────────────────────────────────────

MANUAL_BLOCK_KEYS = {
    '1': 'both_down',
    '2': 'pow',
    '3': 'push',
    '4': 'player_down',
    '5': 'stumble',
}
MANUAL_BLOCK_DISPLAY = {
    'both_down':   'Both Down',
    'pow':         'POW!',
    'push':        'Push',
    'player_down': 'Player Down',
    'stumble':     'Stumble',
}


def _manual_entry(game: GameState,
                  auto_results: list[DieResult] | None = None) -> list[DieResult] | None:
    """
    Console-based manual die entry.
    Returns a list of DieResult objects, or None if cancelled.

    Dice on screen are numbered left-to-right, top-to-bottom (Die 1, Die 2…).
    auto_results shows what was auto-detected so you only need to fix ? ones.

    Commands per die:
      b1-b5   block die  (1=BD 2=POW 3=Push 4=PD 5=Stumble)
      d1-d5   d6 pip face (1-5)
      d6      d6 BB logo face
      8:n     d8 result
      16:n    d16 result
      ok / .  accept the auto-detected result for this die (if not ?)
      done/d  finish — accept remaining dice as auto-detected
      cancel  abort manual entry entirely
    """
    print("\n" + "─" * 50)
    print("  MANUAL ENTRY MODE")
    print("  Look at the camera window — dice are numbered 1, 2, 3…")
    print("  left-to-right, top-to-bottom.")
    print("─" * 50)
    print("  b1=BD  b2=POW  b3=Push  b4=PD  b5=Stumble")
    print("  d1-d5=pip  d6=BB-logo  8:n=d8  16:n=d16")
    print("  ok / .  = accept auto result   done/d = finish   cancel = abort")
    print()

    # Show auto-detected results as reference
    if auto_results:
        print("  Auto-detected:")
        for i, r in enumerate(auto_results):
            flag = "  ✓" if r.raw_value is not None else "  ?"
            print(f"    Die {i+1}: {r.display} ({r.confidence*100:.0f}%){flag}")
        print()

    n_auto = len(auto_results) if auto_results else 0
    results = []

    while True:
        die_num = len(results) + 1

        # Show what auto-detection had for this die
        auto_hint = ""
        if auto_results and die_num <= n_auto:
            ar = auto_results[die_num - 1]
            auto_hint = f" [auto: {ar.display}]" if ar.raw_value is not None else " [auto: ?]"

        try:
            raw = input(f"  Die {die_num}{auto_hint}> ").strip().lower()
        except EOFError:
            return None

        if raw in ('cancel', 'c'):
            print("  Manual entry cancelled.")
            return None

        # done — finish here, fill remaining from auto if available
        if raw in ('done', 'd'):
            if not results:
                print("  No dice entered — cancelled.")
                return None
            # Append remaining auto results if they were good
            if auto_results:
                for ar in auto_results[len(results):]:
                    if ar.raw_value is not None:
                        results.append(ar)
                    else:
                        print(f"  Warning: Die {len(results)+1} was '?' — skipped.")
            break

        # ok / . — accept auto result for this die
        if raw in ('ok', '.'):
            if auto_results and die_num <= n_auto:
                ar = auto_results[die_num - 1]
                if ar.raw_value is not None:
                    results.append(ar)
                    print(f"    → accepted: {ar.display}")
                    continue
                else:
                    print(f"  Auto result was '?' — please enter manually.")
                    continue
            else:
                print(f"  No auto result for Die {die_num} — please enter manually.")
                continue

        # Block die: b1-b5
        if raw.startswith('b') and raw[1:] in MANUAL_BLOCK_KEYS:
            sym = MANUAL_BLOCK_KEYS[raw[1:]]
            results.append(DieResult(
                die_type='block', raw_value=sym,
                display=MANUAL_BLOCK_DISPLAY[sym],
                confidence=1.0, is_numeric=False))
            print(f"    → {MANUAL_BLOCK_DISPLAY[sym]}")
            continue

        # d6 pip: d1-d5
        if raw.startswith('d') and raw[1:] in ('1','2','3','4','5'):
            pip = int(raw[1:])
            results.append(DieResult(
                die_type='d6_bb', raw_value=pip,
                display=str(pip),
                confidence=1.0, is_numeric=True))
            print(f"    → d6 pip {pip}")
            continue

        # d6 BB logo: d6
        if raw == 'd6':
            results.append(DieResult(
                die_type='d6_bb', raw_value=6,
                display='6 (BB)',
                confidence=1.0, is_numeric=True))
            print(f"    → d6 BB Logo")
            continue

        # d8: 8:n
        if raw.startswith('8:'):
            try:
                val = int(raw[2:])
                assert 1 <= val <= 8
                results.append(DieResult(
                    die_type='d8', raw_value=val,
                    display=str(val),
                    confidence=1.0, is_numeric=True))
                print(f"    → d8: {val}")
                continue
            except (ValueError, AssertionError):
                pass

        # d16: 16:n
        if raw.startswith('16:'):
            try:
                val = int(raw[3:])
                assert 1 <= val <= 16
                results.append(DieResult(
                    die_type='d16', raw_value=val,
                    display=str(val),
                    confidence=1.0, is_numeric=True))
                print(f"    → d16: {val}")
                continue
            except (ValueError, AssertionError):
                pass

        print(f"  Unknown: '{raw}'. Use b1-b5, d1-d6, 8:n, 16:n, ok, done, cancel")

    print(f"  Entered {len(results)} dice: {[r.display for r in results]}")
    print("─" * 50)
    return results


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Blood Bowl Dice Tracker")
    ap.add_argument("--camera",  default=0,
                    help="Camera source: 0,1,2 for USB index, or http:// for stream")
    ap.add_argument("--player1", default="Player 1")
    ap.add_argument("--player2", default="Player 2")
    ap.add_argument("--load",    default=None,
                    help="Path to saved session.json to resume")
    return ap.parse_args()


def safe_camera_source(src):
    try:
        return int(src)
    except (ValueError, TypeError):
        return src


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    source = safe_camera_source(args.camera)

    print("=" * 55)
    print("  Blood Bowl Dice Tracker")
    print("=" * 55)

    try:
        cam = Camera(source=source)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("\nAvailable cameras:", Camera.list_usb_cameras())
        sys.exit(1)

    game = GameState(args.player1, args.player2)
    if args.load and os.path.exists(args.load):
        game.load(args.load)
        print(f"Resumed game from {args.load}")
    game.set_active_player(0)

    reader  = DieReader()
    tracker = DiceStabilityTracker()

    tray_roi: tuple | None = None
    tray_check_every = 60   # check every 60 frames — tray doesn't move
    frame_count = 0
    last_detections_raw = []   # hold last detection result for smoothing
    detect_every = 1           # detect every frame for accuracy

    # ── Per-frame state ───────────────────────────────────────────────────────
    last_settled    = False
    flash_counter   = 0
    last_results:   list[DieResult] = []
    last_labels:    list[str]       = []
    last_detections                 = []

    # auto_recorded: True = a roll has been recorded and is waiting for SPACE
    # confirmed:     True = SPACE was pressed this roll cycle; ignore re-settles
    auto_recorded = False
    confirmed     = False   # blocks re-recording after confirm until dice move

    print(f"\nPlayer 1: {args.player1}")
    print(f"Player 2: {args.player2}")
    print("\nControls:")
    print("  SPACE    Confirm roll")
    print("  BKSP     Undo last roll")
    print("  M        Manual entry")
    print("  1 / 2    Switch active player")
    print("  N        Next turn manually")
    print("  S        Save game")
    print("  P        Screenshot (saves screenshot_TIMESTAMP.jpg)")
    print("  Q / ESC  Quit (auto-saves)")
    print("\nWindow opening...\n")

    window_name = "Blood Bowl Dice Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1300, 680)

    while True:
        frame = cam.read()
        if frame is None:
            print("Lost camera feed — retrying...")
            time.sleep(0.1)
            continue

        # ── Tray ROI ──────────────────────────────────────────────────────────
        frame_count += 1
        if frame_count % tray_check_every == 1:
            new_roi = find_tray_roi(frame)
            if new_roi is not None:
                if tray_roi is None:
                    tray_roi = new_roi   # first detection — accept immediately
                else:
                    # Smooth: blend 90% old + 10% new — tray is fixed,
                    # small changes are just detection noise
                    tray_roi = tuple(
                        int(tray_roi[i] * 0.9 + new_roi[i] * 0.1)
                        for i in range(4)
                    )

        # ── Detect ───────────────────────────────────────────────────────────
        # Only re-detect every N frames — reduces flicker from noisy frames
        if frame_count % detect_every == 0:
            last_detections_raw = detect_dice(frame, roi=tray_roi)
        detections = last_detections_raw

        # ── Stability ────────────────────────────────────────────────────────
        settled = tracker.update(detections)

        # If dice moved after a confirm, unlock for next roll
        if confirmed and not settled and len(detections) == 0:
            confirmed = False
            last_detections_raw = []

        # ── Read on first settle only — skip if already confirmed this roll ──
        if settled and not last_settled and not confirmed and not auto_recorded:
            results: list[DieResult] = []
            labels:  list[str]       = []
            for det in detections:
                die_type = classify_die_type(det, frame.shape)
                result   = reader.read(det.crop, die_type)
                results.append(result)
                if result.confidence >= CONFIDENCE_THRESHOLD:
                    labels.append(f"{result.display} {result.confidence*100:.0f}%")
                else:
                    labels.append("?")

            flash_counter   = FLASH_FRAMES
            last_results    = results
            last_labels     = labels
            last_detections = detections
            print(f"[Turn {game.current_turn}] Dice settled: {labels}")
            game.record_roll(results)
            auto_recorded = True

        if flash_counter > 0:
            flash_counter -= 1

        last_settled = settled

        # ── Draw ──────────────────────────────────────────────────────────────
        # Show current detections always — only fall back to last_detections
        # during the flash window immediately after recording a roll
        display_dets   = last_detections if (flash_counter > 0) else detections
        display_labels = last_labels     if (flash_counter > 0) else (last_labels if settled and auto_recorded else [])
        composite = draw_frame(
            camera_frame = frame,
            game         = game,
            detections   = display_dets,
            die_labels   = display_labels,
            settled      = settled,
            flash_green  = (flash_counter > 0),
            tray_roi     = tray_roi,
        )
        cv2.imshow(window_name, composite)

        # ── Keys ──────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):           # Q / ESC — quit
            print("\nQuitting...")
            game.save_all(SAVE_PATH)
            break

        elif key == ord(' '):               # SPACE — confirm roll (stays same player/turn)
            if auto_recorded:
                turn_data = game.confirm_turn()
                tracker.reset(); last_detections_raw = []
                auto_recorded = False
                confirmed     = True
                last_settled  = False
                last_detections = []
                last_labels   = []
                flash_counter = 0
                print(f"[Turn {game.current_turn}] Roll confirmed for "
                      f"{game.players[game.active_player]}. "
                      f"Press 1/2 to switch player, N to advance turn.")
            else:
                print("Nothing to confirm — roll dice first.")

        elif key in (8, 127):               # BACKSPACE — undo
            if game.undo_last_roll():
                auto_recorded = False
                confirmed     = False
                tracker.reset(); last_detections_raw = []
                print("Undone last roll.")
            else:
                print("Nothing to undo.")

        elif key == ord('m'):               # M — manual entry
            # Undo any auto-recorded result first so we start clean
            if auto_recorded:
                game.undo_last_roll()
                auto_recorded = False

            manual = _manual_entry(game, auto_results=last_results or None)
            if manual is not None:
                game.record_roll(manual)
                auto_recorded = True
                last_labels   = [r.display for r in manual]
                last_detections = []
                print(f"[Turn {game.current_turn}] Manual entry recorded: "
                      f"{last_labels}")

        elif key == ord('1'):               # Player 1
            game.set_active_player(0)
            tracker.reset()
            auto_recorded = False
            confirmed     = False
            print(f"Active player: {game.players[0]}")

        elif key == ord('2'):               # Player 2
            game.set_active_player(1)
            tracker.reset()
            auto_recorded = False
            confirmed     = False
            print(f"Active player: {game.players[1]}")

        elif key == ord('n'):               # N — advance turn counter
            if auto_recorded:
                game.confirm_turn()   # lock in any unconfirmed roll first
            game.advance_turn()
            tracker.reset(); last_detections_raw = []
            auto_recorded = False
            confirmed     = False
            last_detections = []
            last_labels   = []
            print(f"Advanced to turn {game.current_turn} "
                  f"(active: {game.players[game.active_player]}).")
            if game.game_over:
                print("\nGame over! 16 turns complete.")
                _show_final_scores(game)
                game.save_all(SAVE_PATH)
                break

        elif key == ord('s'):               # S — save
            game.save_all(SAVE_PATH)
            print("Saved.")

        elif key == ord('p'):               # P — screenshot
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            shot_path = os.path.join(BASE, f"screenshot_{ts}.jpg")
            cv2.imwrite(shot_path, composite)
            print(f"Screenshot saved: screenshot_{ts}.jpg")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def _show_final_scores(game: GameState):
    print("\n" + "=" * 55)
    print("  FINAL SCORES — 16 TURNS COMPLETE")
    print("=" * 55)
    for pi, s in enumerate(game.summaries):
        print(f"\n{s.name}")
        print(f"  d6  total : {s.d6_total}  (BB logo x{s.d6_bb_total})")
        print(f"  d8  total : {s.d8_total}  (avg {s.d8_average():.1f})")
        print(f"  d16 total : {s.d16_total}  (avg {s.d16_average():.1f})")
        print(f"  Block die :")
        for sym, count in s.block_counts.items():
            print(f"    {sym:<16}: {count}")
    print("=" * 55)


if __name__ == "__main__":
    main()

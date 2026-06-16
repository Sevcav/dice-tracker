"""
hw_selftest.py — actively drive the OLEDs and LEDs to verify wiring.

Unlike `python hardware.py` (which only checks libraries + button presses),
this WRITES to both OLEDs and blinks each LED so you can see them work.
Run on the Pi (venv active):  python deploy/hw_selftest.py
"""
import sys
import time

sys.path.insert(0, ".")
from hardware import Hardware, LED_PINS

hw = Hardware()
print(f"GPIO: {hw.available}   OLEDs found: {len(hw._oleds)}")

# 1. LEDs — light each one for 0.6s in turn, then all on, then all off.
print("LED test: each lights in turn (p1, p2, reject, undo)...")
for name in LED_PINS:
    print(f"  -> {name} ON")
    hw.set_led(name, True)
    time.sleep(0.6)
    hw.set_led(name, False)
print("  all LEDs ON for 1.5s")
for name in LED_PINS:
    hw.set_led(name, True)
time.sleep(1.5)
hw.all_leds_off()

# 2. OLEDs — write a distinct message so you can tell them apart / not swapped.
print("OLED test: writing to both displays...")
hw.show("P1", [{"label": "OLED OK", "conf": 99, "stable": True,
                "uncertain": False}], state="selftest")
time.sleep(0.5)
print("Both OLEDs should now show:  Player: P1 selftest / OLED OK 99%")
print("Leaving it on the screens for 8s...")
time.sleep(8)

hw.cleanup()
print("done")

"""
hardware.py
-----------
Physical-rig I/O for the Raspberry Pi: 4 arcade buttons, 4 LEDs, and 2
SSD1309 OLED displays. Imports cleanly on the PC (where gpiozero/luma are
absent) as a no-op stub, so dice_tracker runs unchanged on the bench.

Pin map is the LOCKED allocation from DESIGN.md (BCM numbering), verified
software-clean (no SPI-bus collision, all plain GPIO):

  Buttons (switch -> GND, internal pull-up):
    P1 Confirm  GPIO17    P2 Confirm  GPIO27
    Reject      GPIO22    Undo        GPIO23
  LEDs (GPIO -> LED -> GND, inline resistor already on the LED):
    P1  GPIO5   P2  GPIO6   Reject  GPIO13   Undo  GPIO19
  OLEDs (shared SPI0): SCLK GPIO11, MOSI GPIO10, DC GPIO9, RES GPIO25;
    OLED-1 (P1) CE0/GPIO8, OLED-2 (P2) CE1/GPIO7.

Usage from dice_tracker:
    hw = Hardware(on_event)          # on_event(name) for button presses
    hw.set_led("p1", True)           # name in p1/p2/reject/undo
    hw.show(player, dice_status)     # render the live read to both OLEDs
    hw.available                     # False on the PC stub

Button event names match the keyboard actions already in dice_tracker:
    "p1" / "p2"  -> set active player + confirm (player attribution)
    "reject"     -> reject current settled read
    "undo"       -> undo last logged roll
"""

from __future__ import annotations

# BCM pin map — single source of truth (mirrors DESIGN.md locked table)
BUTTON_PINS = {"p1": 17, "p2": 27, "reject": 22, "undo": 23}
LED_PINS    = {"p1": 5,  "p2": 6,  "reject": 13, "undo": 19}
OLED_PINS   = dict(sclk=11, mosi=10, dc=9, res=25, ce0=8, ce1=7)

# Three-state uncertainty colors are monochrome on OLED: we render a "?"
# suffix for uncertain reads (same semantics as the HUD CONF_UNCERTAIN).

# LED brightness 0.0-1.0 via software PWM (PWMLED works on any GPIO pin).
# The 5mm LEDs at full drive are blinding indoors; 0.2 is a calm tabletop
# glow. Edit this to taste.
LED_BRIGHTNESS = 0.2

try:
    from gpiozero import PWMLED, Button
    _GPIO_OK = True
except Exception:
    _GPIO_OK = False

try:
    from luma.core.interface.serial import spi
    from luma.oled.device import ssd1309
    from PIL import ImageDraw, ImageFont
    _OLED_OK = True
except Exception:
    _OLED_OK = False


class Hardware:
    """Real GPIO/OLED on a Pi; a silent no-op everywhere else."""

    def __init__(self, on_event=None, debounce_s: float = 0.12):
        self.on_event = on_event or (lambda name: None)
        self.available = _GPIO_OK
        self._buttons = {}
        self._leds = {}
        self._oleds = []
        self._font = None

        if _GPIO_OK:
            for name, pin in LED_PINS.items():
                self._leds[name] = PWMLED(pin)   # software PWM, dimmable
            for name, pin in BUTTON_PINS.items():
                b = Button(pin, pull_up=True, bounce_time=debounce_s)
                b.when_pressed = (lambda n=name: self.on_event(n))
                self._buttons[name] = b

        if _OLED_OK:
            try:
                # both displays on SPI0, differentiated by chip-select.
                # RST (GPIO25) is SHARED. Only the FIRST device may drive
                # it; if BOTH pulse RST, building device 1 (P2) resets
                # device 0 (P1) right after it init'd, blanking P1 on the
                # rig. Second device passes gpio_RST=None.
                self._oleds = [
                    ssd1309(spi(port=0, device=0, gpio_DC=OLED_PINS["dc"],
                                gpio_RST=OLED_PINS["res"])),
                    ssd1309(spi(port=0, device=1, gpio_DC=OLED_PINS["dc"],
                                gpio_RST=None)),
                ]
                self._font = ImageFont.load_default()
            except Exception as e:
                print(f"[hardware] OLED init failed: {e}")
                self._oleds = []

    # ── LEDs ────────────────────────────────────────────────────────────
    def set_led(self, name: str, on: bool):
        led = self._leds.get(name)
        if led is not None:
            # PWMLED.value is the duty cycle 0.0-1.0; "on" = dimmed level.
            led.value = LED_BRIGHTNESS if on else 0.0

    def all_leds_off(self):
        for led in self._leds.values():
            led.off()

    # ── OLED render ─────────────────────────────────────────────────────
    def show(self, player: str, dice_status: list[dict], state: str = ""):
        """Mirror the live read to both OLEDs. dice_status items:
        {label, conf, stable, uncertain} — same dicts dice_tracker already
        builds for the phone UI. Both screens show identical content (the
        per-player placement is ergonomic, not different data)."""
        if not self._oleds:
            return
        # After a confirm, stop showing the (now-logged) dice — that reads
        # as if it's still live. Show a clear "logged, ready for next roll"
        # message instead, so the player knows the read was recorded.
        if state == "confirmed":
            lines = ["Last roll CONFIRMED",
                     f"Player {player}",
                     "watching (roll...)"]
        else:
            lines = [f"Player: {player}   {state}"]
            for d in dice_status:
                mark = "?" if d.get("uncertain") else ""
                lines.append(f"  {d['label']}{mark}  {d.get('conf','')}%")
            if not dice_status:
                lines.append("  (roll...)")
        for dev in self._oleds:
            try:
                from luma.core.render import canvas
                with canvas(dev) as draw:
                    y = 0
                    for ln in lines[:6]:
                        draw.text((0, y), ln[:21], fill="white",
                                  font=self._font)
                        y += 11
            except Exception:
                pass

    def cleanup(self):
        self.all_leds_off()
        for dev in self._oleds:
            try:
                dev.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    hw = Hardware(on_event=lambda n: print("button:", n))
    print(f"GPIO available: {_GPIO_OK}   OLED available: {_OLED_OK}")
    print(f"buttons: {BUTTON_PINS}")
    print(f"leds:    {LED_PINS}")
    if hw.available:
        print("Press buttons (Ctrl-C to exit)...")
        import signal
        signal.pause()
    else:
        print("Stub mode (not a Pi) — import works, all calls no-op.")

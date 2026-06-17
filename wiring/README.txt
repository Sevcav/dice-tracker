GPIO wiring diagrams for the BB Dice Tracker (regenerated 2026-06-17).

gpio_wiring_diagram.png  - PRIMARY. Full 40-pin Pi header with the real
                           per-pin pinout (3V3/5V/GND/GPIO labels), plus
                           the components + colored wires.
gpio_wiring_alt.png      - alternate layout, same pin map (compact header).

AUTHORITATIVE SOURCE: DESIGN.md "LOCKED GPIO pin map" table.
The images are PIN-MAP visual aids. Every GPIO/physical-pin pair AND the
40-pin header pinout were verified against the table + the official Pi
pinout 2026-06-17 (no hallucinated pins this time). BUT the images are
NOT literal point-to-point routing -- the drawn wires are decorative;
follow the LABELS and the DESIGN.md table, not the pixel a wire touches.

BUTTON WIRE COLORS (signal wire to GPIO; ground wire = WHITE):
  P1 Confirm = BLUE   -> GPIO17 (Pin 11)
  P2 Confirm = GREEN  -> GPIO27 (Pin 13)
  Reject     = RED    -> GPIO22 (Pin 15)
  Undo       = YELLOW -> GPIO23 (Pin 16)

OLED VCC = 3V3 (Pin 1, never 5V); OLED GND = any real ground pin
(6, 9, 14, 20, 25, 30, 34, 39). OLED bus shared except CS:
OLED-1 CS = GPIO8/CE0 (Pin 24), OLED-2 CS = GPIO7/CE1 (Pin 26).

Pins are software-verified (no SPI-bus collision, all drivable by
gpiozero/RPi.GPIO + luma.oled). Buttons: switch->GND, internal pull-up.
LEDs: red->GPIO, black->GND (inline resistor already present).

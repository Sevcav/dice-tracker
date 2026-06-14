GPIO wiring diagrams for the BB Dice Tracker (generated 2026-06-14).

gpio_wiring_diagram.png  - primary, clearest labels + 40-pin header grid
gpio_wiring_alt.png      - alternate layout, same pin map

AUTHORITATIVE SOURCE: DESIGN.md "LOCKED GPIO pin map" table.
The images are PIN-MAP visual aids. The GPIO/physical-pin pairs for the
buttons, LEDs, and OLED SPI bus were checked against the table, but the
images are NOT literal point-to-point routing -- the drawn wires are
decorative; follow the LABELS and the DESIGN.md table, not the pixel a
wire appears to touch.

KNOWN IMAGE ERROR (gpio_wiring_diagram.png v2): a "GND (Pin 7)" label is
WRONG -- physical pin 7 is GPIO4, NOT a ground. Ignore that label and its
dangling wire. OLED VCC = 3V3 (pin 1, never 5V); OLED GND = any real
ground pin (6, 9, 14, 20, 25, 30, 34, 39).

Pins are also software-verified (no SPI-bus collision, all drivable by
gpiozero/RPi.GPIO + luma.oled). Buttons: switch->GND, internal pull-up.
LEDs: red->GPIO, black->GND (inline resistor already present).

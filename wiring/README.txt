GPIO wiring diagrams for the BB Dice Tracker (generated 2026-06-14).

gpio_wiring_diagram.png  - primary, clearest labels + 40-pin header grid
gpio_wiring_alt.png      - alternate layout, same pin map

AUTHORITATIVE SOURCE: DESIGN.md "LOCKED GPIO pin map" table.
The images are verified-correct PIN MAPS (every GPIO/physical-pin label
checked against the table). They are NOT literal point-to-point routing
diagrams -- the drawn wires are decorative; follow the LABELS, not the
exact pixel a wire appears to touch.

Pins are also software-verified (no SPI-bus collision, all drivable by
gpiozero/RPi.GPIO + luma.oled). Buttons: switch->GND, internal pull-up.
LEDs: red->GPIO, black->GND (inline resistor already present).

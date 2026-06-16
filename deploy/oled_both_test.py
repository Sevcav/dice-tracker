"""
oled_both_test.py — drive BOTH OLEDs at once with DIFFERENT text on each,
so you can see which physical screen is on which chip-select in a single
run. CE0/pin24 shows "P1 / pin24 / CE0"; CE1/pin26 shows "P2 / pin26 /
CE1". Run on the Pi (.venv active):
    python deploy/oled_both_test.py
Ctrl-C to stop.

Expected: LEFT screen -> P1/pin24, RIGHT screen -> P2/pin26. If BOTH
screens show the SAME label (e.g. both "P1"), only one panel is really
responding and the other's CS is shared/floating. If the labels are on
the wrong sides, the two CS wires are swapped.
"""
import time

from luma.core.interface.serial import spi
from luma.core.render import canvas
from luma.oled.device import ssd1309
from PIL import ImageFont

font = ImageFont.load_default()

# Build BOTH displays, each on its own chip-select, shared DC/RST.
# IMPORTANT: RST (GPIO25) is shared. Each ssd1309 constructor pulses RST,
# so if BOTH claim gpio_RST=25 the second build RESETS the first panel
# right after it init'd -> first screen goes blank. Only the FIRST device
# drives the shared reset; the second passes gpio_RST=None.
dev0 = ssd1309(spi(port=0, device=0, gpio_DC=9, gpio_RST=25))    # CE0 / pin24
dev1 = ssd1309(spi(port=0, device=1, gpio_DC=9, gpio_RST=None))  # CE1 / pin26

screens = [
    (dev0, "P1", "pin24", "CE0"),
    (dev1, "P2", "pin26", "CE1"),
]

print("Driving BOTH OLEDs. CE0/pin24 -> P1, CE1/pin26 -> P2. Ctrl-C to stop.")
i = 0
try:
    while True:
        i += 1
        for dev, who, pin, ce in screens:
            with canvas(dev) as draw:
                draw.text((0, 0), who, fill="white", font=font)
                draw.text((0, 14), pin, fill="white", font=font)
                draw.text((0, 28), ce, fill="white", font=font)
                draw.text((0, 42), f"frame {i}", fill="white", font=font)
        time.sleep(0.25)
except KeyboardInterrupt:
    print("\nstopped")

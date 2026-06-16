"""
oled1_test.py — drive ONLY OLED-1 (CS = CE0 / pin 24 / blue wire) in a
loop. Pair with oled2_test.py to confirm which physical screen is which
and that BOTH chip-selects work. Run on the Pi (venv active):
    python deploy/oled1_test.py
Ctrl-C to stop.
"""
import time

from luma.core.interface.serial import spi
from luma.core.render import canvas
from luma.oled.device import ssd1309
from PIL import ImageFont

# OLED-1 only: SPI0 device 0 = CE0 (pin 24), shared DC=GPIO9, RST=GPIO25
dev = ssd1309(spi(port=0, device=0, gpio_DC=9, gpio_RST=25))
font = ImageFont.load_default()

print("Driving OLED-1 (CE0/pin24/blue) in a loop. Ctrl-C to stop.")
i = 0
try:
    while True:
        i += 1
        with canvas(dev) as draw:
            draw.text((0, 0), "OLED-1 (P1)", fill="white", font=font)
            draw.text((0, 14), f"frame {i}", fill="white", font=font)
            draw.text((0, 28), "CS=pin24 CE0", fill="white", font=font)
        time.sleep(0.25)
except KeyboardInterrupt:
    print("\nstopped")

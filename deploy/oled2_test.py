"""
oled2_test.py — drive ONLY OLED-2 (CS = CE1 / pin 26) in a loop so an
intermittent connection can be wiggle-tested. Run on the Pi (venv active):
    python deploy/oled2_test.py
Ctrl-C to stop. While it runs, GENTLY press/wiggle OLED-2's CS (purple)
wire and its splitter junctions; if the text flickers/recovers as you do,
that wire/junction is the loose one.
"""
import time

from luma.core.interface.serial import spi
from luma.core.render import canvas
from luma.oled.device import ssd1309
from PIL import ImageFont

# OLED-2 only: SPI0 device 1 = CE1 (pin 26), shared DC=GPIO9, RST=GPIO25
dev = ssd1309(spi(port=0, device=1, gpio_DC=9, gpio_RST=25))
font = ImageFont.load_default()

print("Driving OLED-2 (CE1/pin26) in a loop. Ctrl-C to stop.")
print("Wiggle the purple CS wire + splitters; watch for flicker/recovery.")
i = 0
try:
    while True:
        i += 1
        with canvas(dev) as draw:
            draw.text((0, 0), "OLED-2 (P2)", fill="white", font=font)
            draw.text((0, 14), f"frame {i}", fill="white", font=font)
            draw.text((0, 28), "CS=pin26 CE1", fill="white", font=font)
        time.sleep(0.25)
except KeyboardInterrupt:
    print("\nstopped")

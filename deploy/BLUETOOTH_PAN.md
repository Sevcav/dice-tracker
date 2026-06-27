# Bluetooth access to the web UI (Surface Pro, no WiFi)

At the store the hotspot/cellular WiFi is unreliable, which breaks the phone
web UI even though the Pi's dice **detection** works fine. This sets up a
**Bluetooth PAN** (Personal Area Network) so the **Surface Pro does exactly what
the phone does over WiFi** — same Align / Live / Games pages, same controls —
just over Bluetooth instead.

**No application code changes.** The Pi runs a Bluetooth NAP (Network Access
Point); the Surface joins as a PAN client. That brings up a normal TCP/IP link
over Bluetooth, so the browser just opens `http://192.168.44.1:5000/` and hits
the *identical* Flask app. The phone-over-WiFi path is untouched and still works
whenever WiFi is available.

```
 Pi (dicetracker)                         Surface Pro (Windows 11)
 ┌───────────────────────────┐           ┌──────────────────────────┐
 │ dice_tracker.py + Flask   │           │ Edge/Chrome browser      │
 │   :5000  (0.0.0.0)        │           │  http://192.168.44.1:5000│
 │ bt-nap.service (NAP)──────┼─Bluetooth─┼──PAN client (Access pt.) │
 │ pan0 bridge 192.168.44.1  │   PAN     │  gets 192.168.44.x (DHCP)│
 │ dnsmasq (DHCP on pan0)    │           │                          │
 └───────────────────────────┘           └──────────────────────────┘
```

The static `192.168.44.1` also kills the "DHCP moves the Pi's IP" gotcha from
HANDOFF — the Bluetooth URL never changes.

---

## One-time Pi setup

Run on the Pi from the repo root (`~/dice-tracker`). All steps are idempotent.

```bash
# 1. Packages: NAP server (bluez-tools), bridge tooling, DHCP.
sudo apt-get update
sudo apt-get install -y bluez-tools bridge-utils dnsmasq

# 2. DHCP for the PAN link.
sudo cp deploy/dnsmasq-pan.conf /etc/dnsmasq.d/pan.conf
sudo systemctl enable --now dnsmasq
sudo systemctl restart dnsmasq

# 3. NAP + bridge service (brings up pan0, then runs bt-network).
chmod +x deploy/pan0.sh
sudo cp deploy/bt-nap.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bt-nap

# 4. Sanity check.
ip addr show pan0                 # should show inet 192.168.44.1/24
systemctl status bt-nap dnsmasq   # both active
```

> If `dnsmasq` fails to start, another resolver may hold port 53 — this config
> sets `port=0` (DHCP only) to avoid that, so a failure here usually means a
> stale system dnsmasq config; check `journalctl -u dnsmasq`.

> Windows-committed `.sh` files sometimes carry CRLF line endings (see HANDOFF).
> If `pan0.sh` won't run, `sed -i 's/\r$//' deploy/pan0.sh`.

---

## One-time pairing (Pi ⇄ Surface)

On the Pi:

```bash
bluetoothctl
# inside the prompt:
power on
agent on
default-agent
discoverable on
pairable on
# ...now pair from the Surface (next section). When it appears / asks to pair,
# accept. Then trust it so it reconnects automatically:
trust <SURFACE_BT_MAC>
discoverable off       # optional once paired
quit
```

On the **Surface Pro**: Settings → Bluetooth & devices → **Add device** →
Bluetooth → pick **dicetracker** → confirm the pairing code matches the Pi.

---

## One-time Surface setup (Windows 11)

After pairing, tell Windows to use the Pi as a network access point:

1. Settings → Bluetooth & devices → **Devices** → scroll to **More devices and
   printer settings** (opens the classic Devices and Printers panel).
2. Right-click **dicetracker** → **Connect using** → **Access point**.
3. Windows brings up the PAN link and pulls a `192.168.44.x` lease from the Pi.

Verify in a terminal (`ipconfig`): the Bluetooth Network Connection should have
an address like `192.168.44.10`. Then:

```
ping 192.168.44.1
```

Open `http://192.168.44.1:5000/` in the browser and bookmark it.

---

## Daily use at the store

1. Pi powered on, `dice_tracker.py` running (or the `dice-tracker` service).
2. On the Surface: **Connect using → Access point** on the **dicetracker**
   device (step 2 above — this is the only recurring step).
3. Browse to `http://192.168.44.1:5000/`.
   - **Align** → match the tray to the green outline, Confirm.
   - **Live** → live read + dice-type / player buttons.
   - **Games** → the BB3 record + CSV export.

Everything works with WiFi/cellular fully off — that's the point.

---

## Troubleshooting

- **Surface got a 169.254.x.x address** (APIPA, not 192.168.44.x): the DHCP
  lease didn't arrive. On the Pi check `systemctl status dnsmasq`, confirm the
  conf is at `/etc/dnsmasq.d/pan.conf`, and that `bnep0` is enslaved to the
  bridge: `bridge link` (or `brctl show pan0`) should list a `bnep*` interface
  once the Surface connects. Reconnect the Access point from Windows.
- **No `pan0` interface**: `sudo systemctl restart bt-nap`, then
  `ip addr show pan0`. Check `journalctl -u bt-nap` — a missing `bt-network`
  means `bluez-tools` didn't install.
- **Can't reach :5000 but ping works**: the tracker isn't running, or it bound
  elsewhere — confirm `dice_tracker.py`/`webapp.py` is up; it listens on
  `0.0.0.0:5000` so the PAN address is covered.
- **Pairing won't take / won't reconnect**: re-pair, and make sure you ran
  `trust <SURFACE_BT_MAC>` in `bluetoothctl` so the Pi accepts it automatically.
- **Alignment video is sluggish over Bluetooth**: expected ceiling — only the
  alignment MJPEG is bandwidth-heavy (live play is tiny JSON polling). It's a
  brief one-time step; if it's too slow we can add a low-quality stream mode
  used only by the Surface (deferred follow-up).

## Fallback

If Bluetooth ever proves too slow, a **USB tether** (Pi ⇄ Surface USB, RNDIS
gadget) gives a faster wired link to the same `:5000` UI — at the cost of a
cable. Not set up here; noted for completeness.

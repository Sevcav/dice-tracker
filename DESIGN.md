# Blood Bowl Dice Tracker — Design Document

**Last updated:** May 2, 2026
**Status:** Production rig CAD in progress

---

## 1. Project Vision

A portable, self-contained device that captures Blood Bowl dice rolls at gaming
stores and tournaments. Players roll into a tray, the camera reads the dice,
the result is attributed to the rolling player via a dedicated button, and the
roll is logged to a database for later analysis via a web app.

**Not a livestream/replay device.** A **data capture device**.

### Core requirements

- Self-contained — no PC on the table
- Works under variable lighting (gaming store conditions)
- Powered by USB power bank for portability
- Modular 3D-printed rig (lower tier + lid + removable dice tray + power bank shelf)
- Phone connects over WiFi for the web UI

---

## 2. Production Rig Design (Active)

The production rig is a **modular two-tier design** with a removable dice tray.

### Architecture

| Tier | Purpose |
|---|---|
| **Lower tier** | Electronics enclosure — Pi, buttons, LEDs, OLEDs, camera mount |
| **Lid** | Sits on top of lower tier, has tray opening + posts hanging down to bolt to lower tier |
| **Dice tray (removable)** | Drops into the lid's tray opening, felt-lined, removable for felt application and Pi access |
| **Power bank shelf** | External, attaches to rear of rig |

### Outer Dimensions

- **Outer footprint:** 260 × 190mm (W × D) with 55 × 55mm chamfered front corners
- **Lower tier height:** 60mm
- **Lid height:** 60mm
- **Total stack height:** 120mm + tray protrusion

### Material

- **PETG** throughout (heat resistance for hot summer venues)
- 4mm wall thickness on all major shells

---

## 3. Component Status

| Component | Status | Notes |
|---|---|---|
| **Lower tier** | ✅ Modeled | All cuts complete: buttons, LEDs, OLED windows, camera mount |
| **Lid (with posts)** | ✅ Modeled | Tray opening + 4 hanging posts with captive M4 nuts |
| **Dice tray (removable)** | ⏳ To design | Felt-lined insert, drops into lid tray opening |
| **Camera base plate** | ⏳ To design | Replaces MakerWorld Camera_Base bottom; bolts into lower tier rear detent |
| **Power bank shelf** | ⏳ To design | External rear shelf for UGREEN power bank |
| **Camera cradle** | 🔧 Blocked | Need Arducam in hand for pocket sizing |
| **Bench prototype tray cradle** | ✅ Printed | Standalone test, validated soft-tray support concept |
| **Bench prototype arm foot** | ✅ Printed | Bridges over cradle back posts; captive nuts; friction-fit |
| **Camera_Base + Camera_Link** | ✅ Printed | From MakerWorld 627829, will be re-used for production rig |

---

## 4. Production Rig — Detailed Specs

### 4.1 Lower Tier

| Spec | Value |
|---|---|
| Outer footprint | 260 × 190mm with 55mm front chamfers |
| Wall thickness | 4mm |
| Height | 60mm |
| Front face: button holes | 4 × Ø27.78mm |
| Front face: LED holes | 4 × Ø8mm (for 7.5mm snap-in bezels) |
| Button positions (X) | ±56.7 and ±18.9mm |
| Button vertical (Z) | ~33mm centered |
| LED positions (X) | Same as buttons |
| LED vertical (Z) | ~9mm |
| Both chamfered corners | OLED window (62.53 × 40.28mm) + PCB recess (3mm deep) |
| Rear face: camera mount detent | 50 × 50 × 3mm pocket on outside |
| Rear inside wall: nut pockets | 4× hex pockets, 0.25mm deep, on inside of rear wall |
| Rear bolt clearance | 4 × Ø4.5mm holes through rear wall |
| Camera mount bolt path | Outside (camera plate) → through 4.5mm clearance → into captive nut on inside |

### 4.2 Lid

| Spec | Value |
|---|---|
| Outer footprint | 251.44 × 181.44mm with chamfered front corners (matches lower tier outline) |
| Wall thickness | 4mm |
| Height | 60mm |
| Tray opening (visible from top) | 160mm wide × 130mm deep |
| Tray opening lip | 4mm wide step around perimeter, 2mm deep recess |
| Tray opening biased toward rear | Yes |
| Posts (4) | Round Ø10mm columns hanging from lid underside |
| Post length | 54mm (engages 2mm into lower tier floor pocket) |
| Post positions | Front: (±110, -30); Rear: (±115, +85) |
| Post captive nut | M4 hex pocket at bottom of each post (3.4mm deep) |
| Post bolt clearance | Ø4.5mm hole through center of each post |
| Bolt entry | From bottom of lower tier (countersunk) |

### 4.3 Dice tray (still to design)

| Spec | Value |
|---|---|
| Outer flange (sits on lid lip) | 159.7 × 138.7mm × 2mm tall |
| Tray body (drops through lid) | 153.7 × 132.7mm |
| Total height | ~52mm |
| Wall thickness | 4mm |
| Floor thickness | 3mm |
| Inner draft | 5° outward (top wider) |
| Felt lining | Self-adhesive felt sheet, applied after print |

### 4.4 Camera base plate (still to design)

| Spec | Value |
|---|---|
| Plate size | ~50 × 50mm (sized to fit in 50×50mm rear detent of lower tier) |
| Plate thickness | 3mm (recessed flush into detent) |
| Top mating | Hinge clevis matching Camera_Link's M4 hinge |
| Bottom mating | 4× M4 clearance holes matching lower tier captive nut pattern |

### 4.5 Power bank shelf (still to design)

| Spec | Value |
|---|---|
| Power bank dimensions | 160.5 × 81 × 26.5mm (UGREEN Nexode) |
| Shelf attachment | Snap-fits or bolts to rear face of lower tier |
| Cable management | Routes USB-C power to rear of rig into Pi |

---

## 5. Detection Software

### Pipeline

1. **Mask** — find dice in tray (HSV color or dark-object mask)
2. **Stability tracker** — wait for dice to settle (no motion)
3. **Crop** — extract per-die crops
4. **CLAHE preprocess** — illumination normalization
5. **CNN classify** — MobileNetV3-Small, exported to ONNX

### Dice scope

| Die | Detection | Classes |
|---|---|---|
| Block dice (cream) | ✅ CNN | 5 (POW, Push, Both Down, Player Down, Stumble) |
| BB d6 (black) | ✅ CNN | 6 (1-5, BB Logo) |
| D16 (cream, trapezohedron) | 🔄 Future CNN | 16 (1-16) — for injury rolls |
| D8 (scatter) | 📱 Manual entry on phone | — |
| D3 | 📱 Manual entry on phone | — |

**Note:** Camera angle changed from overhead to ~35° forward bank shot. The
existing CNN was trained on overhead images and will need retraining once we
capture training data from the new angle.

### Player attribution

- 4 buttons on rig: **P1 Confirm**, **P2 Confirm**, **Reject**, **Undo**
- Whoever pressed **their** confirm button = the rolling player
- No turn tracking on rig — that lives in the web app
- Phone web app selects dice type (Block / D6 / D16) before each roll

### Database schema (planned)

```
rolls table:
  id              auto-increment
  game_id         FK
  player          P1 or P2
  dice_type       "block" | "d6" | "d16" | "d8" | "d3"
  dice_results    JSON list e.g. ["POW", "Push", "POW"]
  timestamp       unix epoch
  rejected        bool (if REJECT was pressed before confirm)
  raw_image_path  optional, for review
```

---

## 6. Hardware

### On hand

- Raspberry Pi 4B in TH3D aluminum case (91 × 65 × 33mm)
- Pi 3B (spare)
- Pi Camera v2.1 with 22" ribbon (CSI port non-functional on this Pi 4 — moved to USB path)
- USB camera (test only, not the production camera)
- Bambu P1S 3D printer
- M3 + M4 hardware in stock (M4 lengths: 8/12/16/20mm)
- M3 + M4 nuts in stock
- 32GB microSD (current, getting full)
- HiLetgo SPI 2.42" OLED 128×64 (verified dimensions): PCB 71×43mm, glass 62.25×40mm, header zone 14mm at top

### On order / shipping

- 64GB A2 microSD
- Arducam 1080P Day/Night IR USB camera (OV2710 sensor)
- Bambu LED Lamp Kit 001 ×2 (USB 5V, built-in PC diffuser)
- WMYCONGCONG arcade buttons (verified: thread Ø26.25mm, dial Ø33mm, body length 62mm — using Ø27.78mm holes for print clearance)
- 5mm pre-wired LEDs in 7.5mm snap-in bezels (using Ø8mm holes)
- UGREEN Power Bank (160.5 × 81 × 26.5mm)
- Pre-crimped JST pigtails
- M4 brass heat-set inserts (production future use, not yet needed)

---

## 7. Software State

### Pi-side stack (validated working)

- Python 3.13.5
- OpenCV 4.10.0
- ONNX Runtime 1.25.0
- Flask 3.1.1

### Code repo

- GitHub: github.com/Sevcav/dice-tracker
- Pi clones from this repo
- ONNX model + labels.json transferred via SCP (binary, not committed to git)

### Networking

- iPhone hotspot saved as `iphone-hotspot` connection (for game stores)
- Home WiFi (Knickerbocker on Deco mesh) saved separately as `home-knickerbocker`
- NetworkManager (`nmcli`) controls WiFi on Bookworm/Trixie
- mDNS broadcasting needed for `dicetracker.local` access from phone (TODO)
- Pi default IP on iPhone hotspot: 172.20.10.x (Apple subnet)
- Pi default IP on home network: 192.168.68.88 (DHCP, may shift)

### Tested

- ✅ Pi 4 boots, SSH works
- ✅ All Python libraries import cleanly
- ✅ ONNX classifier loads with all 11 classes
- ✅ USB camera capture (test cam, just for proof)
- ✅ iPhone hotspot connection saved
- ✅ Home WiFi connection saved

### Not yet started

- Flask web UI scaffold
- Database schema implementation
- D16 training data capture
- D16 CNN training
- Retraining CNN on new ~35° camera angle
- mDNS broadcast for `dicetracker.local`
- Wiring + GPIO code for buttons / OLEDs / LEDs

---

## 8. Open Questions / Deferred Decisions

| Topic | Status |
|---|---|
| Lighting strategy | IR camera handles dark; lamps optional; lamps ordered for testing |
| Camera angle | ~35° forward bank shot from rear-center — confirmed |
| D16 detection | Deferred until block + d6 working end-to-end |
| OLED retention method | Friction / hot glue / M2 screws — TBD when OLEDs in hand |
| Phone web UI scaffold | Not started |
| Database schema | Drafted, not implemented |
| Camera cradle for Arducam | Blocked — need camera in hand for measurements |
| Cable routing strategy | External clips for prototype, internal channels for production |
| Aesthetic direction | Utilitarian for prototype; organic curves for production v2 |
| Power bank shelf attachment method | Snap-fit, screw, or magnetic — TBD |

---

## 9. Known Lessons / Misconceptions

- **Cradle ≠ production base.** The 200×200 cradle was a standalone bench
  test, not the foundation of the full rig.
- **Pi camera ribbon hardware appears dead** on the current Pi 4 — using USB
  camera path instead. Will not pursue ribbon further.
- **PyTorch is too big for the Pi.** ONNX Runtime is the runtime path; PyTorch
  stays on Windows for training only.
- **Apostrophes and `$` in WiFi credentials** broke `nmcli` — needed single
  quotes in shell, and full key-mgmt declaration.
- **wpa_supplicant.conf is ignored on Bookworm/Trixie** — NetworkManager is
  in charge. All WiFi config goes through `nmcli`.
- **STL external data** — PyTorch ONNX export creates a `.onnx.data` sidecar
  for large models. Both files must be transferred together.
- **Deco mesh routers can block new devices** — single SSID for both bands
  caused initial connectivity issues; ethernet was the most reliable path
  for initial setup.
- **Captive nuts beat heat-set inserts** when inserts haven't shipped yet,
  and captive nuts are stronger anyway (steel vs brass).
- **Print pause at the right layer** is critical for captive nuts — too late
  and the pocket caps over; too early and pocket walls haven't formed yet.
- **OLED PCB has a header strip** at the top — total height includes 14mm
  header zone above the active glass area. Account for this when sizing
  display windows.
- **OLED requires inside-out mounting** — PCB sits on inside of shell wall,
  glass protrudes outward through the through-window. Solder pins on PCB
  back protrude 2mm and need clearance.
- **Bumped lower tier height to 60mm** because OLED at 39.78mm tall plus
  4mm floor + 4mm top wouldn't fit in original 50mm box.
- **PCB recess depth dictates wall thickness** — initially considered 6mm
  walls for outside pocket, but moving to inside-mounting kept walls at 4mm.
- **OLED through-window goes glass-only** — header strip stays inside the
  box, only the glass + active display protrudes through the window cut.

---

## 10. Decisions Locked

### Hardware geometry
- ✅ Lower tier 260 × 190mm with 55mm front chamfers, 60mm tall, 4mm walls
- ✅ Lid matches lower tier outer footprint, 60mm tall, 4mm walls
- ✅ Tray opening in lid: 160 × 130mm with 2mm × 3-4mm wide perimeter lip
- ✅ Lid mounted to lower tier with 4× Ø10mm posts hanging down + M4 captive nuts + bolts from below
- ✅ Removable dice tray drops into lid pocket (felt-lined)
- ✅ Camera base plate replaces MakerWorld Camera_Base; mounts in 50×50mm rear detent
- ✅ Camera arm: re-use Camera_Link from MakerWorld 627829
- ✅ Power bank external on rear shelf

### Detection
- ✅ Block dice + BB d6 + D16 (CNN); D8/D3 manual entry
- ✅ D16 needed for injury rolls (not kickoff)
- ✅ No turn tracking on rig — pure data capture
- ✅ Camera angle ~35° forward bank shot
- ✅ Production rig direction: utilitarian for prototype, organic v2 later

### Bench prototype (validation only — not production)
- ✅ Tray cradle 200×200×5mm with 4 corner posts (15×15×50mm)
- ✅ Arm foot bridges back two posts of cradle (230 × 40 × 15mm)
- ✅ Both validated by printing

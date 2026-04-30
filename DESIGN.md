# Blood Bowl Dice Tracker — Design Document

**Last updated:** April 30, 2026
**Status:** Pre-production prototype phase

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
- Single 3D-printed shell (or modular assembly)
- Phone connects over WiFi for the web UI

---

## 2. Final Form Factor (Production Target)

Reference: Gemini render with corner OLEDs, sloped front face, rear cantilever
camera arm, accessible Pi/power bank compartment.

**Footprint:** ~250 × 280mm
**Print volume:** Fits Bambu P1S (256³) — assembled from multiple printed parts
**Material:** PETG (heat resistance for hot summer venues)

### Major components

| Component | Spec |
|---|---|
| Base plate with integrated tray pocket | ~250 × 200mm × 25mm |
| Soft-tray corner posts (4) | 15×15×50mm at 90mm offsets |
| Front sloped button panel | 4 arcade buttons, 15° slope |
| Front-corner OLED mounts (2) | 45° outward angle |
| Rear access compartment | Houses Pi + power bank |
| Camera arm | Cantilevered, 35° down look angle |

---

## 3. Bench Prototype (Current Phase)

**Goal:** Validate individual components before committing to the integrated
production base. Print small, test, iterate.

### Validation pieces

| Piece | Status | Notes |
|---|---|---|
| Tray cradle (corner posts) | ✅ Printed, validated | 200×200×5mm plate, 50mm posts |
| Camera arm — Camera_Base | ✅ Printed (from MakerWorld 627829) | Was for V2.1 ribbon cam |
| Camera arm — Camera_Link | ✅ Printed (from MakerWorld 627829) | 137mm forward reach |
| Arm foot (standalone) | 🔧 To design | Replaces Pi-case-as-foot from MakerWorld |
| Camera cradle (USB) | 🔧 To design | Replaces Pi-camera-mount from MakerWorld; needs Arducam in hand first |

### Bench prototype layout

The cradle and arm sit **separately on the table**:

```
   [TRAY CRADLE]            [STANDALONE ARM FOOT]
   (200×200 plate     +     (~150×100 weighted base
    with corner             with arm bolted on top,
    posts holding            camera reaches forward
    the soft tray)           over the tray)
```

This is **NOT the production layout**. This is just for testing the camera +
detection pipeline before we commit to the integrated rig.

---

## 4. Detection Software

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

### Player attribution

- 4 buttons on rig: **P1 Confirm**, **P2 Confirm**, **Reject**, **Undo**
- Whoever pressed **their** confirm button = the rolling player
- No turn tracking on rig — that lives in the web app

---

## 5. Hardware (Pi-Side)

### On hand (already)

- Raspberry Pi 4B in TH3D aluminum case (91 × 65 × 33mm)
- Pi 3B (spare)
- Pi Camera v2.1 with 22" ribbon (ribbon CSI port appears non-functional on this Pi 4 — confirmed via missing dmesg entries; using USB instead)
- USB camera (test only — not the production camera)
- Bambu P1S 3D printer
- M3 + M4 hardware in stock
- 32GB microSD (current, full from initial setup)

### On order

- 64GB A2 microSD (clean install when arrives)
- Arducam IR Day/Night USB camera (1080p, OV2710)
- Bambu LED Lamp Kit 001 ×2 (USB 5V, built-in PC diffuser)
- WMYCONGCONG arcade buttons ×N
- HiLetgo SPI OLED 2.42" displays ×2
- UGREEN Power Bank
- Pre-crimped JST pigtails

---

## 6. Software State

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
- Home WiFi (Knickerbocker on Deco mesh) saved separately
- mDNS broadcasting needed for `dicetracker.local` access from phone (TODO)

---

## 7. Open Questions / Deferred Decisions

| Topic | Status |
|---|---|
| Lighting strategy | IR camera handles dark; lamps optional; lamps ordered for testing |
| Camera angle (overhead vs. side bank-shot) | Decided: ~35° forward bank shot from rear-center |
| D16 detection | Deferred until block + d6 working end-to-end |
| Production rig print strategy | Modular pieces vs. single print — TBD |
| Pi case integration | Keep TH3D aluminum, design compartment around it — TBD |
| OLED corner angle | 45° outward — confirmed |
| Button arrangement | 4 across the front sloped face, P1 leftmost / P2 rightmost — confirmed |
| Phone web UI scaffold | Not started |
| Database schema | Not started |
| Arm foot design | In progress — standalone weighted base for bench prototype |
| Camera cradle for Arducam | Blocked — need camera in hand for measurements |

---

## 8. Known Misconceptions / Lessons

- **Cradle ≠ production base.** The 200×200 cradle we printed is a standalone
  bench test, not the foundation of the full rig.
- **Pi camera ribbon hardware appears dead** on the current Pi 4 — switching
  to USB camera path. Will not pursue ribbon further.
- **PyTorch is too big for the Pi.** ONNX Runtime is the runtime path; PyTorch
  stays on Windows for training only.
- **Apostrophes and `$` in WiFi credentials** broke `nmcli` — needed single
  quotes in shell, and full key-mgmt declaration.
- **wpa_supplicant.conf is ignored on Bookworm/Trixie** — NetworkManager is
  in charge. All WiFi config goes through `nmcli`.
- **STL external data** — PyTorch ONNX export creates a `.onnx.data` sidecar
  for large models. Both files must be transferred together.

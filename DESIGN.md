# Blood Bowl Dice Tracker — Design Document

**Last updated:** April 30, 2026 (evening)
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
| Tray cradle (corner posts) | ✅ Printed, validated | 200×200×5mm plate, 50mm posts at ±90mm offset |
| Camera arm — Camera_Base | ✅ Printed (from MakerWorld 627829) | Was originally for V2.1 ribbon cam |
| Camera arm — Camera_Link | ✅ Printed (from MakerWorld 627829) | 137mm forward reach |
| Arm foot — bridges over back two cradle posts | 🟡 Printing now | Captive M4 nuts, friction-fits over posts |
| Camera cradle (USB Arducam) | 🔧 Blocked | Need camera in hand for pocket sizing |

### Bench prototype layout (revised)

The arm foot **bridges across the two back posts** of the cradle. No separate base — the cradle posts themselves are the structural anchors for the camera arm.

```
        [Camera @ end of Camera_Link]
                       ↓
              [Camera_Base, M4 bolted]
                       ↓
              [ARM FOOT — bridges back two posts]
            ┌─────┐                    ┌─────┐
            │     │                    │     │
            │ POST│                    │ POST│   ← back two posts
            │     │                    │     │     of the cradle
   ┌────────┴─────┴────────────────────┴─────┴────────┐
   │                                                  │
   │                   TRAY (in pocket)               │
   │                                                  │
   └──────┬─────┬─────────────────────┬─────┬─────────┘
          │POST │                     │POST │   ← front two posts
          └─────┘                     └─────┘
```

This is the **bench prototype layout**, not the production form factor.

### Arm Foot Spec (bench prototype)

- **Bridge:** 230 × 40 × 15mm (PETG, 30% gyroid)
- **Sockets:** 27×27mm outer, 16×16mm inner pocket, 35mm tall, 5.5mm walls
- **Socket centers:** ±97.5mm from foot center (matches cradle back post positions)
- **Mounting:** Friction-fit over posts (no bolts to cradle)
- **Camera_Base mount:** 4× M4 holes in 59.2 × 16.8mm pattern, centered on bridge
- **Captive nut method:** M4 hex pockets (7.2mm flat-to-flat × 3.4mm deep) embedded mid-thickness; print pause at layer ~34 to drop in 4 nuts before resuming
- **Bolt spec:** M4 × 12mm flat head, 4 each (Camera_Base to foot)
- **Through-cut top:** 16×16 squares cut all the way through bridge so posts pass through entirely

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

## 5. Hardware (Pi-Side)

### On hand (already)

- Raspberry Pi 4B in TH3D aluminum case (91 × 65 × 33mm)
- Pi 3B (spare)
- Pi Camera v2.1 with 22" ribbon (ribbon CSI port appears non-functional on this Pi 4 — confirmed via missing dmesg entries; using USB path instead)
- USB camera (test only — not the production camera)
- Bambu P1S 3D printer
- M3 + M4 hardware in stock (M4 lengths: 8/12/16/20mm)
- M3 + M4 nuts in stock
- 32GB microSD (current, full from initial setup)

### On order

- 64GB A2 microSD (clean install when arrives)
- Arducam 1080P Day/Night IR USB camera (OV2710 sensor)
- Bambu LED Lamp Kit 001 ×2 (USB 5V, built-in PC diffuser)
- WMYCONGCONG arcade buttons ×N
- HiLetgo SPI OLED 2.42" displays ×2
- UGREEN Power Bank
- Pre-crimped JST pigtails
- M4 brass heat-set inserts (for production rig — not used on bench prototype)

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
- Home WiFi (Knickerbocker on Deco mesh) saved separately as `home-knickerbocker`
- 2.4GHz forced via `freq_list` in original wpa_supplicant attempts (not used; NetworkManager controls WiFi on Bookworm/Trixie)
- mDNS broadcasting needed for `dicetracker.local` access from phone (TODO)
- Pi default IP on iPhone hotspot: 172.20.10.x (Apple subnet)
- Pi default IP on home network: 192.168.68.88 (DHCP, may shift)

### Tested

- ✅ Pi 4 boots, SSH works
- ✅ All Python libraries import cleanly
- ✅ ONNX classifier loads with all 11 classes
- ✅ USB camera capture (low quality test cam, just for proof)
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

## 7. Open Questions / Deferred Decisions

| Topic | Status |
|---|---|
| Lighting strategy | IR camera handles dark; lamps optional; lamps ordered for testing |
| Camera angle | ~35° forward bank shot from rear-center — confirmed |
| D16 detection | Deferred until block + d6 working end-to-end |
| Production rig print strategy | Modular pieces vs. single print — TBD |
| Pi case integration | Keep TH3D aluminum, design compartment around it — TBD |
| OLED corner angle | 45° outward — confirmed |
| Button arrangement | 4 across the front sloped face, P1 leftmost / P2 rightmost — confirmed |
| Phone web UI scaffold | Not started |
| Database schema | Drafted, not implemented |
| Arm foot design | ✅ Designed, printing now |
| Camera cradle for Arducam | Blocked — need camera in hand for measurements |
| Production rig CAD | Deferred until bench prototype validates full pipeline |
| Cable routing strategy | External clips for prototype, internal channels for production |
| Aesthetic direction | Utilitarian for prototype; organic curves for production v2 |

---

## 8. Known Misconceptions / Lessons

- **Cradle ≠ production base.** The 200×200 cradle we printed is a standalone
  bench test, not the foundation of the full rig. The arm foot bridges over
  the cradle's two back posts as a temporary mounting solution.
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
- **Deco mesh routers can block new devices** — the WiFi was failing because
  TP-Link Deco doesn't broadcast separate 2.4/5GHz SSIDs by default; even
  with that fixed, NetworkManager + ethernet was the most reliable path for
  initial setup.
- **Captive nuts beat heat-set inserts** for this use case because the inserts
  hadn't shipped yet, and captive nuts are stronger anyway (steel vs brass).
- **Print pause at the right layer** is critical for captive nuts — too late
  and the pocket caps over before the pause; too early and the pocket walls
  haven't formed yet. ~34 layers at 0.2mm for our M4 hex pockets.
- **Bridge length must accommodate sockets, not just bolt patterns.** Initial
  220mm bridge length didn't leave room for 27×27 sockets at ±97.5mm. Bumped
  to 230mm.
- **Print tolerance for posts:** 0.28mm typical printer tolerance; 16mm pocket
  on 15mm post leaves ~0.2mm slop per side after print — snug, no wobble.

---

## 9. Decisions Locked This Session

- ✅ Bench prototype: arm foot bridges over the cradle back posts (no separate base)
- ✅ Captive M4 nuts in foot for Camera_Base mounting
- ✅ Through-cut sockets so posts pass all the way through the foot
- ✅ Friction-fit foot over posts (no bolts into cradle)
- ✅ Foot dimensions: 230 × 40 × 15mm bridge, 27×27 sockets, 35mm walls
- ✅ Detection scope: Block dice + BB d6 + D16 (CNN); D8/D3 manual entry
- ✅ D16 needed for injury rolls, not kickoff events
- ✅ No turn tracking on rig — pure data capture
- ✅ Camera angle: ~35° forward bank shot from rear-center mount
- ✅ Production rig direction: utilitarian first, organic v2 later

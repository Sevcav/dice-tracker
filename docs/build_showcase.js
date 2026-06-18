/*
 * build_showcase.js — generates docs/BB-Dice-Tracker-Showcase.docx
 * A styled, printable showcase of the Blood Bowl Dice Tracker with the
 * real images embedded (concept, assembled rig, wiring, Fusion model,
 * Games page, Live page) and a concept-to-working-assembly writeup.
 *
 * Images are read from this docs/ folder and auto-sized to fit the page
 * (capped by width AND height so portrait shots don't overflow).
 *
 * Run:  node docs/build_showcase.js
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  PageBreak, Footer, PageNumber, VerticalAlign,
} = require("docx");

const DIR = __dirname;
const OUT = path.join(DIR, "BB-Dice-Tracker-Showcase.docx");

// ── Palette (Blood Bowl-ish: deep navy, slate blue, red, gold) ──
const NAVY = "1D3557";
const SLATE = "457B9D";
const RED = "E63946";
const GOLD = "E9C46A";
const LIGHT = "F1F5F9";
const INK = "222222";
const MUTE = "5F6368";

// Page geometry (US Letter, 1" margins)
const CONTENT_W = 9360;            // DXA
const MAX_IMG_W_IN = 6.0;          // inches
const MAX_IMG_H_IN = 7.0;          // inches (keeps tall shots on one page)
const PX = 96;                     // px per inch for ImageRun transformation

// Compute fit size (px) for an image given its pixel dims.
function fitPx(w, h) {
  const r = w / h;
  let wIn = MAX_IMG_W_IN;
  let hIn = wIn / r;
  if (hIn > MAX_IMG_H_IN) { hIn = MAX_IMG_H_IN; wIn = hIn * r; }
  return { width: Math.round(wIn * PX), height: Math.round(hIn * PX) };
}

const { ImageRun } = require("docx");
const { execSync } = require("child_process");

// Read an image and its dimensions (PNG/JPEG header parse, no deps).
function imgMeta(file) {
  const data = fs.readFileSync(path.join(DIR, file));
  let w, h;
  if (data[0] === 0x89 && data[1] === 0x50) { // PNG
    w = data.readUInt32BE(16); h = data.readUInt32BE(20);
  } else { // JPEG: scan SOF markers
    let i = 2;
    while (i < data.length) {
      if (data[i] !== 0xFF) { i++; continue; }
      const m = data[i + 1];
      if (m >= 0xC0 && m <= 0xCF && m !== 0xC4 && m !== 0xC8 && m !== 0xCC) {
        h = data.readUInt16BE(i + 5); w = data.readUInt16BE(i + 7); break;
      }
      i += 2 + data.readUInt16BE(i + 2);
    }
  }
  const ext = file.toLowerCase().endsWith(".png") ? "png" : "jpg";
  return { data, w, h, ext };
}

// A framed, centered, auto-sized figure with a styled caption. The frame
// is a single-cell table (cell borders avoid the w:pBdr ordering quirk).
function figure(file, captionText) {
  const m = imgMeta(file);
  const sz = fitPx(m.w, m.h);
  const fb = { style: BorderStyle.SINGLE, size: 8, color: SLATE };
  return [
    new Table({
      alignment: AlignmentType.CENTER,
      width: { size: CONTENT_W, type: WidthType.DXA },
      columnWidths: [CONTENT_W],
      rows: [new TableRow({ children: [new TableCell({
        width: { size: CONTENT_W, type: WidthType.DXA },
        borders: { top: fb, left: fb, bottom: fb, right: fb },
        margins: { top: 80, bottom: 80, left: 80, right: 80 },
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 40, after: 40 },
          children: [new ImageRun({
            type: m.ext, data: m.data,
            transformation: { width: sz.width, height: sz.height },
            altText: { title: captionText, description: captionText, name: file },
          })],
        })],
      })] })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 80, after: 240 },
      children: [new TextRun({ text: captionText, italics: true, size: 18,
        color: MUTE })],
    }),
  ];
}

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1,
    children: [new TextRun(text)] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2,
    children: [new TextRun(text)] });
}
function body(text) {
  return new Paragraph({ spacing: { after: 160 }, alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, size: 22, color: INK })] });
}

// A tinted callout box (single-cell table) for the intro.
function callout(lines) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: CONTENT_W, type: WidthType.DXA },
      shading: { fill: LIGHT, type: ShadingType.CLEAR },
      borders: {
        top: { style: BorderStyle.SINGLE, size: 2, color: LIGHT },
        left: { style: BorderStyle.SINGLE, size: 24, color: RED },
        bottom: { style: BorderStyle.SINGLE, size: 2, color: LIGHT },
        right: { style: BorderStyle.SINGLE, size: 2, color: LIGHT },
      },
      margins: { top: 160, bottom: 160, left: 240, right: 200 },
      children: lines.map((t, i) => new Paragraph({
        spacing: { after: i === lines.length - 1 ? 0 : 120 },
        alignment: AlignmentType.JUSTIFIED,
        children: [new TextRun({ text: t, size: 22, color: INK })],
      })),
    })] })],
  });
}

// A styled BOM table: a navy section-header row spanning both columns,
// a column-header row, then zebra-striped Component / Notes rows.
function bomTable(title, rows) {
  const COL1 = 3400, COL2 = CONTENT_W - COL1;
  const cell = (text, opts = {}) => new TableCell({
    width: { size: opts.w, type: WidthType.DXA },
    columnSpan: opts.span,
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ children: [new TextRun({
      text, bold: !!opts.bold, size: opts.size || 20,
      color: opts.color || INK,
    })] })],
  });
  const thin = { style: BorderStyle.SINGLE, size: 2, color: "D0D5DD" };
  const dataRows = rows.map((r, i) => new TableRow({ children: [
    cell(r[0], { w: COL1, bold: true, fill: i % 2 ? "FFFFFF" : LIGHT, color: NAVY }),
    cell(r[1], { w: COL2, fill: i % 2 ? "FFFFFF" : LIGHT }),
  ] }));
  return [new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [COL1, COL2],
    borders: { top: thin, left: thin, bottom: thin, right: thin,
      insideHorizontal: thin, insideVertical: thin },
    rows: [
      new TableRow({ children: [
        cell(title, { w: CONTENT_W, span: 2, bold: true, size: 22,
          color: "FFFFFF", fill: NAVY }),
      ] }),
      new TableRow({ tableHeader: true, children: [
        cell("Component", { w: COL1, bold: true, color: "FFFFFF", fill: SLATE }),
        cell("Notes", { w: COL2, bold: true, color: "FFFFFF", fill: SLATE }),
      ] }),
      ...dataRows,
    ],
  })];
}

// A phased step: bold lead-in + text, with a gold accent.
function step(lead, text) {
  return new Paragraph({
    spacing: { after: 140 },
    alignment: AlignmentType.JUSTIFIED,
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: GOLD, space: 10 } },
    indent: { left: 200 },
    children: [
      new TextRun({ text: lead + "  ", bold: true, size: 22, color: NAVY }),
      new TextRun({ text, size: 22, color: INK }),
    ],
  });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 30, bold: true, font: "Calibri", color: NAVY },
        paragraph: { spacing: { before: 300, after: 60 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: GOLD, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 24, bold: true, font: "Calibri", color: SLATE },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1440, bottom: 1080, left: 1440 },
      },
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        border: { top: { style: BorderStyle.SINGLE, size: 4, color: SLATE, space: 6 } },
        children: [new TextRun({ text: "Blood Bowl Dice Tracker   •   Page ",
          size: 16, color: MUTE }),
          new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUTE })],
      })] }),
    },
    children: [
      // ── Title banner (full-width navy table) ──
      new Table({
        width: { size: CONTENT_W, type: WidthType.DXA },
        columnWidths: [CONTENT_W],
        rows: [new TableRow({ children: [new TableCell({
          width: { size: CONTENT_W, type: WidthType.DXA },
          shading: { fill: NAVY, type: ShadingType.CLEAR },
          borders: {
            top: { style: BorderStyle.NONE },
            left: { style: BorderStyle.NONE },
            bottom: { style: BorderStyle.SINGLE, size: 24, color: RED },
            right: { style: BorderStyle.NONE },
          },
          margins: { top: 280, bottom: 280, left: 240, right: 240 },
          verticalAlign: VerticalAlign.CENTER,
          children: [
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
              children: [new TextRun({ text: "BLOOD BOWL DICE TRACKER",
                bold: true, size: 44, color: "FFFFFF" })] }),
            new Paragraph({ alignment: AlignmentType.CENTER,
              children: [new TextRun({
                text: "A portable, camera-based dice-roll capture rig for tournaments",
                italics: true, size: 22, color: GOLD })] }),
          ],
        })] })],
      }),
      new Paragraph({ spacing: { after: 200 }, children: [] }),

      // ── What it is (callout) ──
      callout([
        "The Blood Bowl Dice Tracker is a self-contained, portable rig that "
        + "automatically reads and records every dice roll in a game. A camera "
        + "on an overhead arm watches a felt dice tray; an on-board Raspberry "
        + "Pi runs a trained computer-vision model that identifies each die the "
        + "instant the roll settles.",
        "The read is shown on twin player-facing OLED screens and logged to a "
        + "phone-friendly web app — no laptop, no manual tallying, and no "
        + "arguments about what was actually rolled. It keeps a per-player, "
        + "per-face dice record for the whole game, and pre-fills the coaches’ "
        + "names straight from the league schedule.",
      ]),

      // ── Concept ──
      h1("The Concept"),
      body("The idea began as a single rendering: a clean, arcade-style box "
        + "with a felt tray, an overhead camera arm, and four big buttons — "
        + "something that would sit on a tournament table and simply work."),
      ...figure("Concept.png",
        "Image 1 — The original concept rendering: overhead camera arm, felt "
        + "tray, and a four-button control deck."),

      new Paragraph({ children: [new PageBreak()] }),

      // ── Hardware ──
      h1("From Concept to Working Hardware"),
      body("The printed-and-assembled rig follows the concept closely: a 3D-"
        + "printed case houses the Raspberry Pi, the four arcade buttons, and "
        + "the player OLEDs, with a removable felt tray and an adjustable arm "
        + "carrying the IR camera overhead. The camera runs in forced infrared "
        + "mode — a printed lens cap darkens the sensor’s photoresistor so it "
        + "always sees the lighting the model was trained on."),
      ...figure("Completed Rig.jpg",
        "Image 2 — The printed and assembled rig: case, four arcade buttons, "
        + "felt tray, and the overhead IR camera on its adjustable arm."),

      new Paragraph({ children: [new PageBreak()] }),

      // ── Wiring + Fusion ──
      h1("Under the Hood"),
      h2("Wiring"),
      body("Four buttons, four status LEDs, and two SPI OLED displays connect "
        + "directly to the Raspberry Pi’s GPIO header against a locked, "
        + "software-verified pin map — no SPI-bus collisions, internal pull-ups "
        + "for the buttons, and the two OLEDs sharing one bus by chip-select."),
      ...figure("Wiring.png",
        "Image 3 — GPIO wiring: buttons, LEDs, and the dual-OLED SPI bus on "
        + "the Raspberry Pi header."),

      new Paragraph({ children: [new PageBreak()] }),

      h2("The CAD Model (Fusion 360)"),
      body("The case, tray, Pi cage, camera back-plate, and the IR lens cap "
        + "were all designed in Fusion 360 as a one-piece printable enclosure "
        + "with captive-nut pockets, OLED windows, and button cut-outs sized to "
        + "the real components."),
      ...figure("Fusion of Dice Tracker.jpg",
        "Image 4 — The Fusion 360 design: one-piece case with Pi cage, tray "
        + "opening, OLED windows, and button cut-outs."),

      new Paragraph({ children: [new PageBreak()] }),

      // ── Software ──
      h1("The Software"),
      h2("The Games Page — the dice record"),
      body("The rig is headless — the phone (or the on-board screen) is the "
        + "only display. The Games page pins the latest roll for quick "
        + "correction, then shows the running dice record: a per-player tally of "
        + "every face rolled across the whole game. Player names import straight "
        + "from the league’s Game Sheets / TourPlay schedule."),
      ...figure("Games Page.png",
        "Image 5 — The Games page: pinned, editable latest roll above the "
        + "per-player cumulative dice record."),

      new Paragraph({ children: [new PageBreak()] }),

      h2("The Live Page — reading in real time"),
      body("During play, the Live page shows the current read with per-die "
        + "confidence, the active player, and the recent roll log. The same read "
        + "mirrors to the player-facing OLEDs; a single button press confirms "
        + "and logs it, and the screen resets to “watching” for the next roll."),
      ...figure("Live Tracker.jpg",
        "Image 6 — The Live page: current read with confidence, active "
        + "player, dice-type selector, and recent rolls."),

      new Paragraph({ children: [new PageBreak()] }),

      // ── Bill of Materials ──
      h1("Bill of Materials"),
      body("The components that make up the rig as actually built. The "
        + "Raspberry Pi runs everything off USB power; the felt tray, case, "
        + "and IR lens cap are 3D-printed."),
      ...bomTable("Core components (used in the rig)", [
        ["Raspberry Pi 4B", "In TH3D aluminum case (91 × 65 × 33 mm)"],
        ["64 GB A2 microSD", "Boots the Pi; holds models + game database"],
        ["Arducam 1080P Day/Night IR USB camera", "OV2710 sensor; forced IR via a printed lens cap"],
        ["4× WMYCONGCONG arcade buttons", "P1 / P2 confirm, reject, undo"],
        ["4× 5 mm pre-wired LEDs (in bezels)", "Status LEDs; inline resistor on the red lead"],
        ["2× HiLetgo 2.42\" SPI OLED 128×64", "SSD1309; player-facing read displays"],
        ["UGREEN power bank", "Portable power (160.5 × 81 × 26.5 mm)"],
        ["Dupont F-F jumper wires", "OLED / button / LED GPIO wiring"],
        ["1-to-2 wire splitters", "Ground consolidation to the Pi GND pins"],
        ["M5×25 SHCS + M5 nyloc nuts", "Camera-arm friction joints"],
        ["M3 / M4 screws + nuts", "Case assembly"],
      ]),
      new Paragraph({ spacing: { after: 120 }, children: [] }),
      ...bomTable("Optional add-on", [
        ["7\" HDMI touchscreen", "On-rig display; handy but not required (rig runs phone-only)"],
      ]),
      new Paragraph({ spacing: { after: 120 }, children: [] }),
      ...bomTable("Ordered but not used", [
        ["M4 brass heat-set inserts", "Held for a future production revision"],
        ["Bambu LED Lamp Kit ×2", "For the abandoned day-mode lighting; rig is always-IR"],
        ["JST pigtails (pre-crimped)", "Replaced by Dupont jumpers + splitters"],
      ]),

      new Paragraph({ children: [new PageBreak()] }),

      // ── The journey ──
      h1("From Concept to Working Assembly"),
      body("The project moved from a single concept rendering to a fully working "
        + "tournament rig through five distinct phases:"),
      step("Design.", "The case, tray, Pi cage, and IR lens cap were modeled in "
        + "Fusion 360 as a one-piece printable enclosure, sized around the real "
        + "Raspberry Pi, OLEDs, and arcade buttons."),
      step("Vision.", "An early rule-based detector proved too brittle across "
        + "lighting and camera changes, so the system pivoted to a trained YOLO "
        + "object-detection model reading block dice, D6, and D16 from infrared "
        + "frames — with a settle-and-smooth stack so a read only locks once "
        + "the dice have actually come to rest."),
      step("Hardware.", "Buttons, LEDs, and twin OLEDs were wired to the Pi "
        + "against a locked pin map and driven by a clean GPIO layer; the camera "
        + "was forced into infrared mode with a printed lens cap so it always "
        + "matches the trained conditions."),
      step("Portability.", "The whole stack was ported to run on the Pi off a "
        + "lightweight, torch-free inference backend, started automatically at "
        + "boot and controlled entirely from a phone — so the rig works "
        + "anywhere with just power and a phone hotspot."),
      step("Integration.", "The final step connected the rig to the league: "
        + "coach and team names pull directly from the existing Blood Bowl Game "
        + "Sheets / TourPlay schedule, and every confirmed roll is captured into "
        + "a per-player dice record, exportable as CSV."),
      new Paragraph({
        spacing: { before: 200, after: 0 },
        alignment: AlignmentType.CENTER,
        border: {
          top: { style: BorderStyle.SINGLE, size: 8, color: RED, space: 8 },
          bottom: { style: BorderStyle.SINGLE, size: 8, color: RED, space: 8 },
        },
        children: [new TextRun({
          text: "Drop it on the table, power it up, connect a phone — and the "
            + "rig reads, displays, and records every roll of the game.",
          bold: true, size: 24, color: NAVY, italics: true,
        })],
      }),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("Wrote", OUT, "(" + buf.length + " bytes)");
});

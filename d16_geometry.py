"""
d16_geometry.py
---------------
D16 adjacency-deduction layer: geometric cross-checking of d16 face
reads using the physical structure of the die.

THE TABLE (derived 2026-06-12 from 458 labeled dice by
training/derive_d16_adjacency.py, all 16 legal pairs seen 24-33 times,
zero structural conflicts):

  The D16 trapezohedron is two rings of 8 kite faces in CONSECUTIVE
  number order: ring A = 1..8, ring B = 9..16. A resting die shows a
  top face N flanked by its ring neighbors N-1 and N+1 (wrapping within
  the ring), e.g. top 1 shows sides 2 and 8; top 9 shows sides 10 and
  16. Opposite faces (a, 17-a) live in different rings and are never
  co-visible. Both of the user's D16s (white + black, same manufacturer)
  share this layout.

  PHYSICALLY VERIFIED by the user against the real dice on 2026-06-12
  (rolled spot-checks: neighbors always result +/-1 with ring wrap).
  ADJACENCY_VERIFIED is True: impossible-triple flagging AND low-
  confidence top-face deduction are both active in dice_tracker.py.

Used by dice_tracker.py on settled d16 rolls to (a) deduce the top face
when the model's top confidence is low, (b) flag geometrically
impossible face combinations as misreads before the player confirms.
"""

from __future__ import annotations

import re

ADJACENCY_VERIFIED = True    # user-verified against physical dice 2026-06-12

_RINGS = (list(range(1, 9)), list(range(9, 17)))


def _ring_of(face: int) -> list[int] | None:
    for ring in _RINGS:
        if face in ring:
            return ring
    return None


def flanks(top: int) -> tuple[int, int]:
    """The two side faces visible when `top` is up."""
    ring = _ring_of(top)
    i = ring.index(top)
    return (ring[(i - 1) % 8], ring[(i + 1) % 8])


# (side_lo, side_hi) -> top, e.g. (2, 8) -> 1, (10, 16) -> 9
PAIR_TO_TOP: dict[tuple[int, int], int] = {
    tuple(sorted(flanks(t))): t for ring in _RINGS for t in ring
}


def deduce_top(side_a: int, side_b: int) -> int | None:
    """Top face implied by two flanking sides, or None if the pair is
    not a legal flanking pair."""
    return PAIR_TO_TOP.get(tuple(sorted((side_a, side_b))))


def ring_distance(a: int, b: int) -> int | None:
    """Steps around the ring between two faces; None if different rings."""
    ring = _ring_of(a)
    if ring is None or b not in ring:
        return None
    d = abs(ring.index(a) - ring.index(b))
    return min(d, 8 - d)


def covisible(a: int, b: int) -> bool:
    """Can faces a and b appear on the same resting die? True only for
    same-ring faces 1 or 2 steps apart (top+side, or side+side)."""
    d = ring_distance(a, b)
    return d in (1, 2)


def face_value(label: str) -> int | None:
    """'D16_12' -> 12 (None for non-d16 labels)."""
    m = re.fullmatch(r"D16_(\d+)", label)
    return int(m.group(1)) if m else None


def cluster_faces(boxes: list[list[float]], linkage_px: float = 55.0
                  ) -> list[list[int]]:
    """Group face-box indices into per-die clusters by center distance
    (glyphs of one die are ~35-45px apart at 1280x720; different dice
    are further). Same linkage as the training-side clustering."""
    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            if (dx * dx + dy * dy) ** 0.5 < linkage_px:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def analyze_die(faces: list[tuple[int, float]]) -> dict:
    """Geometric verdict for ONE die.

    faces: [(face_value, confidence)] for the die's detected face boxes,
    ordered with the geometric TOP candidate first (caller determines
    the top: the box whose center is nearest the midpoint of the other
    two, same convention the adjacency table was mined with; with two
    boxes pass the higher-confidence first).

    Returns dict(status, top, deduced_top, note):
      status 'ok'         — read is geometrically consistent
             'deduced'    — top deduced/corrected from the sides
                            (only when ADJACENCY_VERIFIED)
             'impossible' — faces cannot coexist on a resting die: misread
             'unknown'    — not enough information to judge
    """
    vals = [v for v, _ in faces]
    if len(faces) == 3:
        top, (a, _ca), (b, _cb) = vals[0], faces[1], faces[2]
        expect = deduce_top(a, b)
        if expect == top:
            return {"status": "ok", "top": top, "deduced_top": top,
                    "note": ""}
        if expect is None:
            return {"status": "impossible", "top": top, "deduced_top": None,
                    "note": f"sides {a},{b} are not a legal pair"}
        if ADJACENCY_VERIFIED:
            return {"status": "deduced", "top": expect, "deduced_top": expect,
                    "note": f"sides {a},{b} imply top {expect}, "
                            f"model read {top}"}
        return {"status": "impossible", "top": top, "deduced_top": expect,
                "note": f"sides {a},{b} imply top {expect}, "
                        f"model read {top}"}
    if len(faces) == 2:
        a, b = vals
        if not covisible(a, b):
            return {"status": "impossible", "top": a, "deduced_top": None,
                    "note": f"faces {a},{b} cannot both be visible"}
        expect = PAIR_TO_TOP.get(tuple(sorted((a, b))))
        if expect is not None and ADJACENCY_VERIFIED:
            # two flanking sides, top box missing -> we know the top
            return {"status": "deduced", "top": expect,
                    "deduced_top": expect,
                    "note": f"sides {a},{b} imply unseen top {expect}"}
        return {"status": "ok", "top": a, "deduced_top": expect, "note": ""}
    if len(faces) == 1:
        return {"status": "ok", "top": vals[0], "deduced_top": None,
                "note": ""}
    return {"status": "unknown", "top": None, "deduced_top": None,
            "note": f"{len(faces)} face boxes on one die"}


def analyze_roll(labels: list[str], boxes: list[list[float]],
                 confidences: list[float]) -> list[dict]:
    """Cluster a settled d16 read into dice and analyze each.
    Returns one verdict dict per die (adds 'indices': the box indices of
    that die, top candidate first)."""
    idx = [i for i, lab in enumerate(labels)
           if face_value(lab) is not None]
    verdicts = []
    for cluster in cluster_faces([boxes[i] for i in idx]):
        gi = [idx[k] for k in cluster]
        centers = [((boxes[i][0] + boxes[i][2]) / 2,
                    (boxes[i][1] + boxes[i][3]) / 2) for i in gi]
        if len(gi) == 3:
            # top = box nearest the midpoint of the other two
            def mid_dist(k: int) -> float:
                others = [m for m in range(3) if m != k]
                mx = (centers[others[0]][0] + centers[others[1]][0]) / 2
                my = (centers[others[0]][1] + centers[others[1]][1]) / 2
                return ((centers[k][0] - mx) ** 2
                        + (centers[k][1] - my) ** 2) ** 0.5
            order = sorted(range(3), key=mid_dist)
            gi = [gi[k] for k in order]
        else:
            gi = sorted(gi, key=lambda i: -confidences[i])
        faces = [(face_value(labels[i]), confidences[i]) for i in gi]
        v = analyze_die(faces)
        v["indices"] = gi
        verdicts.append(v)
    return verdicts


# ── Physical verification record ─────────────────────────────────────────────
# Verified by the user 2026-06-12 with rolled spot-checks on the physical
# dice: the two readable neighbors are always the result +/-1 within its
# ring (1..8 / 9..16, wrapping), matching this table. If a different D16
# (other manufacturer) is ever used, rerun the check — layouts are not
# standardized; a mismatching die needs its own table.

if __name__ == "__main__":
    print("D16 adjacency table (sides -> top):")
    for (a, b), t in sorted(PAIR_TO_TOP.items(), key=lambda kv: kv[1]):
        print(f"  top {t:>2}  <-  sides {a:>2}, {b:>2}")
    print(f"\nADJACENCY_VERIFIED = {ADJACENCY_VERIFIED}")

"""
derive_d16_adjacency.py
-----------------------
Mine the D16 adjacency table from the labeled capture frames.

Every labeled D16 die is three glyph circles: the top face plus the two
visible flanking faces, arranged in an arc. The TOP glyph is the middle
of the arc (closest to the midpoint of the other two). Accumulating
(side_A, side_B) -> top votes over every die in every frame yields the
physical adjacency map — IF the labels are consistent, each unordered
side pair maps to exactly one top.

The derived table is a CANDIDATE: per the work-stream plan it must be
verified against the physical dice by the user before the deduction
layer trusts it (printed checklist at the end).

Output: training/synth_assets/d16_adjacency.json + console report.
"""

import json
from collections import Counter, defaultdict

import numpy as np

from crop_common import (ASSETS, originals_index, read_polygons_raw,
                         source_classes)


def face_num(names, cls) -> int:
    return int(names[cls].split("_")[1])


def main():
    names = source_classes("d16")
    index = originals_index()

    triples = []          # (top, side_a, side_b, stem) per observed die
    for split in ["train", "valid", "test"]:
        for entry in index["d16"][split]:
            polys = read_polygons_raw(entry["label"])
            centers = [p.mean(axis=0) for _, p in polys]
            # cluster glyphs into dice (same linkage as synth_dice)
            parent = list(range(len(polys)))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            for i in range(len(polys)):
                for j in range(i + 1, len(polys)):
                    if np.hypot(*(centers[i] - centers[j])) < 55:
                        parent[find(i)] = find(j)
            groups = defaultdict(list)
            for i in range(len(polys)):
                groups[find(i)].append(i)

            for g in groups.values():
                if len(g) != 3:
                    continue
                # top = glyph closest to the midpoint of the other two
                best, best_d = None, 1e9
                for k in g:
                    others = [m for m in g if m != k]
                    mid = (centers[others[0]] + centers[others[1]]) / 2
                    d = float(np.hypot(*(centers[k] - mid)))
                    if d < best_d:
                        best_d, best = d, k
                sides = sorted(face_num(names, polys[m][0])
                               for m in g if m != best)
                triples.append((face_num(names, polys[best][0]),
                                sides[0], sides[1], entry["stem"]))

    print(f"observed dice: {len(triples)}")

    # consistency: each unordered side pair should name ONE top
    votes: dict[tuple[int, int], Counter] = defaultdict(Counter)
    stems: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for top, a, b, stem in triples:
        votes[(a, b)][top] += 1
        stems[(top, a, b)].append(stem)

    print("pair -> top (votes):")
    table = {}
    conflicts = []
    for pair, cnt in sorted(votes.items()):
        (top1, n1), *rest = cnt.most_common()
        flag = ""
        if rest:
            conflicts.append((pair, dict(cnt)))
            flag = f"  CONFLICT {dict(cnt)}"
        elif n1 <= 2:
            flag = "  (low votes — possible label error)"
        print(f"  sides {pair[0]:>2},{pair[1]:>2} -> top {top1:>2} "
              f"x{n1}{flag}")
        table[pair] = (top1, n1)

    # a triple is suspect if its pair is low-vote or it lost a conflict
    suspect = []
    for (top, a, b), ss in sorted(stems.items()):
        best_top, n1 = table[(a, b)]
        if top != best_top or n1 <= 2:
            suspect.append(((top, a, b), ss))
    if suspect:
        print("\nSUSPECT label triples (check these frames in Roboflow):")
        for (top, a, b), ss in suspect:
            print(f"  top {top} sides {a},{b}: {', '.join(ss)}")

    # adjacency graph: faces co-visible with each face
    adj = defaultdict(set)
    for top, a, b, _ in triples:
        adj[top].update([a, b])
        adj[a].update([top, b])
        adj[b].update([top, a])
    print("\nco-visibility graph (face: seen-with):")
    for f in sorted(adj):
        print(f"  {f:>2}: {sorted(adj[f])}")

    # sum-17 sanity: opposite faces (a, 17-a) must NEVER be co-visible
    bad = [(f, 17 - f) for f in adj if (17 - f) in adj[f]]
    print(f"\nopposite-faces-co-visible violations: {bad or 'none'}")

    out = {
        "observed_dice": len(triples),
        "pair_to_top": {f"{a},{b}": t for (a, b), (t, n) in table.items()},
        "pair_votes":  {f"{a},{b}": n for (a, b), (t, n) in table.items()},
        "conflicts": [{"pair": list(p), "votes": c} for p, c in conflicts],
        "suspect_triples": [{"top": t, "sides": [a, b], "frames": ss}
                            for (t, a, b), ss in suspect],
        "covisibility": {str(f): sorted(adj[f]) for f in sorted(adj)},
    }
    path = ASSETS / "d16_adjacency.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")
    print("\nVERIFY AGAINST THE PHYSICAL DICE before trusting the table:")
    print("  for each row, set the die so the two side numbers face you;")
    print("  the printed top must match. Spot-check at least 5 rows +")
    print("  any conflicted pairs.")


if __name__ == "__main__":
    main()

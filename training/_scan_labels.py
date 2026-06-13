import glob

bad_range, bad_size, n_boxes = [], [], 0
for f in glob.glob(r"C:\Users\chapm\Dice Code\training\datasets"
                   r"\combined_crop\*\labels\*.txt"):
    for ln, line in enumerate(open(f)):
        v = line.split()
        if len(v) != 5:
            continue
        n_boxes += 1
        cx, cy, w, h = map(float, v[1:])
        if not (0 <= cx - w / 2 and cx + w / 2 <= 1
                and 0 <= cy - h / 2 and cy + h / 2 <= 1):
            bad_range.append((f, ln, round(cx - w / 2, 4),
                              round(cx + w / 2, 4), round(cy - h / 2, 4),
                              round(cy + h / 2, 4)))
        if w * 423 < 3 or h * 372 < 3:
            bad_size.append((f, ln, round(w * 423, 1), round(h * 372, 1)))

print(f"total boxes: {n_boxes}")
print(f"out of [0,1] range: {len(bad_range)}")
for b in bad_range[:10]:
    print("  ", b)
print(f"sliver (<3px): {len(bad_size)}")
for b in bad_size[:10]:
    print("  ", b)

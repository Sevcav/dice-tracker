"""
find_holes.py
-------------
Find probable bolt-hole locations in each STL by detecting clusters
of vertices arranged in circular patterns.

Run:
    cd "C:/Users/chapm/Dice Code/Stls"
    python find_holes.py
"""

import os
import struct
import glob
import numpy as np
from collections import defaultdict


def read_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        n_tri = struct.unpack("<I", f.read(4))[0]
        verts = np.zeros((n_tri * 3, 3), dtype=np.float32)
        for i in range(n_tri):
            f.read(12)
            for v in range(3):
                xyz = struct.unpack("<3f", f.read(12))
                verts[i * 3 + v] = xyz
            f.read(2)
        return verts


def find_holes_axis(verts, slice_axis, other_axes, name, axis_name):
    """Slice perpendicular to slice_axis, look for circles in the other 2 axes."""
    a_min = verts[:, slice_axis].min()
    a_max = verts[:, slice_axis].max()
    found_any = False
    for a_target in np.linspace(a_min + 1, a_max - 1, 10):
        slice_verts = verts[np.abs(verts[:, slice_axis] - a_target) < 0.5]
        if len(slice_verts) < 20:
            continue
        xy = slice_verts[:, other_axes]
        clusters = []
        used = np.zeros(len(xy), dtype=bool)
        for i, p in enumerate(xy):
            if used[i]:
                continue
            dists = np.linalg.norm(xy - p, axis=1)
            mask = (dists < 5.0) & ~used
            if mask.sum() >= 8:
                cluster = xy[mask]
                used |= mask
                center = cluster.mean(axis=0)
                radii = np.linalg.norm(cluster - center, axis=1)
                if radii.std() < 0.8 and 0.8 < radii.mean() < 4.0:
                    diameter = radii.mean() * 2
                    clusters.append((center[0], center[1], a_target, diameter))
        unique = []
        for c in clusters:
            is_dup = False
            for u in unique:
                if abs(c[0] - u[0]) < 2 and abs(c[1] - u[1]) < 2:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(c)
        if unique:
            if not found_any:
                print(f"  --- Slicing along {axis_name} ---")
                found_any = True
            print(f"  At {axis_name}={a_target:6.1f}:")
            for x, y, z, d in unique:
                bolt = "M2" if d < 2.5 else ("M3" if d < 3.5 else "M4")
                ax_labels = ["XYZ"[i] for i in range(3) if i != slice_axis]
                print(f"    ({ax_labels[0]}={x:6.1f}, {ax_labels[1]}={y:6.1f})  Ø{d:.2f}mm   {bolt}")


def find_holes(verts, name):
    """
    Find circular hole patterns by slicing along each axis.
    """
    print(f"\n=== {name} ===")
    print(f"  Bounds: X[{verts[:,0].min():.1f},{verts[:,0].max():.1f}]  "
          f"Y[{verts[:,1].min():.1f},{verts[:,1].max():.1f}]  "
          f"Z[{verts[:,2].min():.1f},{verts[:,2].max():.1f}]")
    find_holes_axis(verts, 0, [1, 2], name, "X")
    find_holes_axis(verts, 1, [0, 2], name, "Y")
    find_holes_axis(verts, 2, [0, 1], name, "Z")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for stl_path in sorted(glob.glob(os.path.join(here, "*.STL"))):
        verts = read_stl(stl_path)
        find_holes(verts, os.path.basename(stl_path))

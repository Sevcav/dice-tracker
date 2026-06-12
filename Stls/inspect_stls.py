"""
inspect_stls.py
---------------
Reads each STL file and reports its overall bounding box dimensions
so we know the size of each printed part without opening Fusion.

Run:
    cd "C:/Users/chapm/Dice Code/Stls"
    python inspect_stls.py
"""

import os
import struct
import glob
import numpy as np


def read_stl(path):
    """Read a binary STL and return all vertex points as a numpy array."""
    with open(path, "rb") as f:
        header = f.read(80)            # 80-byte header
        n_tri  = struct.unpack("<I", f.read(4))[0]

        # Each triangle: normal (3 floats) + 3 vertices (9 floats) + attribute (2 bytes) = 50 bytes
        verts = np.zeros((n_tri * 3, 3), dtype=np.float32)
        for i in range(n_tri):
            f.read(12)                          # normal — skip
            for v in range(3):
                xyz = struct.unpack("<3f", f.read(12))
                verts[i * 3 + v] = xyz
            f.read(2)                           # attribute byte count
        return verts


def inspect(path):
    name = os.path.basename(path)
    try:
        v = read_stl(path)
    except Exception as e:
        print(f"  {name}: ERROR — {e}")
        return

    mins = v.min(axis=0)
    maxs = v.max(axis=0)
    size = maxs - mins
    print(f"\n{name}")
    print(f"  Triangles : {len(v)//3}")
    print(f"  X range   : {mins[0]:7.2f}  to  {maxs[0]:7.2f}   ({size[0]:6.2f} mm)")
    print(f"  Y range   : {mins[1]:7.2f}  to  {maxs[1]:7.2f}   ({size[1]:6.2f} mm)")
    print(f"  Z range   : {mins[2]:7.2f}  to  {maxs[2]:7.2f}   ({size[2]:6.2f} mm)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print(f"Inspecting STLs in: {here}\n")
    for stl_path in sorted(glob.glob(os.path.join(here, "*.STL"))):
        inspect(stl_path)

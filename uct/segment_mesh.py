"""
Segment a mesh into disconnected parts and save each as a separate STL.

Each connected component (a set of faces with no shared edges to other faces)
is saved as its own STL file at its original position in world space.

Usage:
    python segment_mesh.py bone.stl
    python segment_mesh.py bone.stl -o parts/
    python segment_mesh.py bone.stl --min-faces 500   # discard tiny fragments
    python segment_mesh.py bone.stl --top 5           # keep only 5 largest parts

Dependencies:
    pip install trimesh numpy
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import trimesh


def segment_mesh(
    input_path: str,
    output_dir: str | None = None,
    min_faces: int = 0,
    top_n: int | None = None,
) -> None:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    out_dir = Path(output_dir) if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    print(f"Loading {input_path} ...")
    mesh = trimesh.load(str(input_path), force="mesh", process=False)
    print(f"  Vertices: {len(mesh.vertices):,}   Faces: {len(mesh.faces):,}")

    # STL files store each triangle independently with no shared vertices.
    # Merge duplicate vertices so adjacency/connectivity can be computed.
    print("  Merging duplicate vertices ...")
    mesh.merge_vertices()
    print(f"  After merge: {len(mesh.vertices):,} vertices")

    # --- split into connected components ------------------------------------
    print("\nFinding connected components ...")
    components = mesh.split(only_watertight=False)
    print(f"  Found {len(components)} component(s)")

    # sort largest first
    components = sorted(components, key=lambda m: len(m.faces), reverse=True)

    # filter by minimum face count
    if min_faces > 0:
        before = len(components)
        components = [m for m in components if len(m.faces) >= min_faces]
        discarded = before - len(components)
        if discarded:
            print(f"  Discarded {discarded} fragment(s) with < {min_faces} faces")

    # keep only top N
    if top_n is not None:
        components = components[:top_n]

    if not components:
        print("No components remaining after filtering.")
        return

    # --- save each component -----------------------------------------------
    n_digits = len(str(len(components)))
    print(f"\nSaving {len(components)} part(s) to {out_dir}/")

    for i, part in enumerate(components):
        out_path = out_dir / f"{stem}_part{str(i + 1).zfill(n_digits)}.stl"
        part.export(str(out_path))
        size_mb = out_path.stat().st_size / 1e6
        print(f"  [{i+1:>{n_digits}}] {len(part.faces):>10,} faces  →  {out_path.name}  ({size_mb:.1f} MB)")

    print(f"\nDone. {len(components)} parts saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Split a mesh into disconnected parts and save each as STL."
    )
    parser.add_argument("input", help="Input STL file")
    parser.add_argument(
        "-o", "--output-dir",
        help="Directory for output STL files (default: same directory as input)",
    )
    parser.add_argument(
        "--min-faces", type=int, default=0, metavar="N",
        help="Discard parts with fewer than N faces (default: 0 = keep all)",
    )
    parser.add_argument(
        "--top", type=int, default=None, metavar="N",
        help="Keep only the N largest parts (default: keep all)",
    )

    args = parser.parse_args()

    segment_mesh(
        args.input,
        output_dir=args.output_dir,
        min_faces=args.min_faces,
        top_n=args.top,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        print("Usage: python segment_mesh.py <mesh.stl> [options]")
        print("       python segment_mesh.py --help")
        sys.exit(0)
    main()

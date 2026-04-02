#!/usr/bin/env python3
"""
Compute per-residue RMSF from an MD trajectory.

By default, computes RMSF of Cα atoms per residue after aligning the protein
to the first frame (to remove overall rotation/translation).

Outputs:
  - <prefix>_rmsf.csv  (resid, resname, rmsf_A)
  - <prefix>_rmsf.npy  (rmsf array in Angstrom)
  - <prefix>_rmsf.png  (optional plot)

Dependencies:
  python3 -m pip install MDAnalysis numpy
  python3 -m pip install matplotlib   # only if --plot
"""

from __future__ import annotations

import argparse
import numpy as np

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.transformations import fit_rot_trans


def main() -> None:
    p = argparse.ArgumentParser(description="Compute per-residue RMSF (default: protein Cα).")
    p.add_argument(
        "-t",
        "--topology",
        required=True,
        help="Topology file (e.g. .pdb or .gro; must match trajectory atom count/order)",
    )
    p.add_argument("-x", "--trajectory", required=True, help="Trajectory file (e.g. .xtc)")
    p.add_argument(
        "--align-selection",
        default="protein and backbone",
        help="Selection used for alignment (default: 'protein and backbone')",
    )
    p.add_argument(
        "--rmsf-selection",
        default="protein and name CA",
        help="Selection for RMSF atoms (default: 'protein and name CA')",
    )
    p.add_argument("--stride", type=int, default=1, help="Frame stride (default: 1)")
    p.add_argument("--out-prefix", default="rmsf", help="Output prefix (default: rmsf)")
    p.add_argument("--plot", action="store_true", help="Save RMSF plot to <prefix>_rmsf.png")
    p.add_argument(
        "--no-pbc-fix",
        action="store_true",
        help="Disable PBC nojump handling (default: try to apply nojump to protein selection)",
    )
    args = p.parse_args()

    u = mda.Universe(args.topology, args.trajectory)
    align_ag = u.select_atoms(args.align_selection)
    rmsf_ag = u.select_atoms(args.rmsf_selection)

    if align_ag.n_atoms == 0:
        raise SystemExit(f"Alignment selection matched 0 atoms: {args.align_selection!r}")
    if rmsf_ag.n_atoms == 0:
        raise SystemExit(f"RMSF selection matched 0 atoms: {args.rmsf_selection!r}")

    # Reference = first frame coordinates (static copy)
    ref = mda.Universe(args.topology, args.trajectory)
    ref.trajectory[0]
    ref_align_ag = ref.select_atoms(args.align_selection)

    if ref_align_ag.n_atoms != align_ag.n_atoms:
        raise SystemExit(
            "Alignment selections differ between mobile and reference; check your selections."
        )

    # Build a transformation pipeline:
    # - Optional PBC "nojump" correction on protein atoms (avoids box-jump artifacts).
    # - Fit rotation/translation to the reference based on align-selection.
    transforms = []
    if not args.no_pbc_fix:
        try:
            from MDAnalysis.transformations import NoJump

            transforms.append(NoJump(align_ag))
        except Exception:
            # NoJump may be unavailable in some MDAnalysis versions; proceed without it.
            pass
    transforms.append(fit_rot_trans(align_ag, ref_align_ag))
    u.trajectory.add_transformations(*transforms)

    # Streaming RMSF (Welford) on RMSF selection; avoids storing all frames.
    n_atoms = rmsf_ag.n_atoms
    mean_pos = np.zeros((n_atoms, 3), dtype=np.float64)
    m2 = np.zeros((n_atoms, 3), dtype=np.float64)
    n = 0
    for ts in u.trajectory[:: args.stride]:
        n += 1
        x = rmsf_ag.positions.astype(np.float64, copy=False)
        delta = x - mean_pos
        mean_pos += delta / n
        delta2 = x - mean_pos
        m2 += delta * delta2

    if n < 2:
        raise SystemExit("Need at least 2 frames to compute RMSF.")

    var = m2 / (n - 1)
    rmsf_A = np.sqrt(var.sum(axis=1))

    # Map atoms -> residues (assumes one atom per residue for default CA selection)
    residues = rmsf_ag.residues
    if len(rmsf_A) == len(residues):
        resid = residues.resids
        resname = residues.resnames
        per_res_rmsf = rmsf_A
    else:
        # General case: average over atoms within each residue
        per_res_rmsf = []
        resid = []
        resname = []
        for r in residues:
            atom_idx = np.isin(rmsf_ag.atoms.indices, r.atoms.indices)
            if not np.any(atom_idx):
                continue
            per_res_rmsf.append(float(np.mean(rmsf_A[atom_idx])))
            resid.append(int(r.resid))
            resname.append(str(r.resname))
        per_res_rmsf = np.asarray(per_res_rmsf, dtype=np.float64)
        resid = np.asarray(resid, dtype=np.int32)
        resname = np.asarray(resname, dtype=object)

    np.save(f"{args.out_prefix}_rmsf.npy", per_res_rmsf)

    # CSV
    out_csv = f"{args.out_prefix}_rmsf.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("resid,resname,rmsf_A\n")
        for i in range(len(per_res_rmsf)):
            f.write(f"{int(resid[i])},{resname[i]},{per_res_rmsf[i]:.6f}\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {args.out_prefix}_rmsf.npy")
    print(f"Residues: {len(per_res_rmsf)}  (selection: {args.rmsf_selection!r})")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as e:
            raise SystemExit(
                "matplotlib is required for plotting. Install with: python3 -m pip install matplotlib"
            ) from e

        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.plot(resid, per_res_rmsf, lw=1.2)
        ax.set_xlabel("Residue id")
        ax.set_ylabel("RMSF (Å)")
        ax.set_title("Per-residue RMSF")
        ax.grid(True, alpha=0.25)
        out_png = f"{args.out_prefix}_rmsf.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()


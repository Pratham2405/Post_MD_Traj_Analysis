#!/usr/bin/env python3
"""
Compute protein backbone dihedral features from a topology + XTC trajectory.

Outputs:
  - <prefix>_raw.npz:
      times_ps, phi, psi, omega, angles_stacked, and basic metadata
  - <prefix>_features.npy:
      sin/cos transformed features suitable for PCA/TICA

Install:
  python3 -m pip install MDAnalysis numpy
"""

from __future__ import annotations

import argparse
import numpy as np

import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_dihedrals


def _hstack_nonempty(arrs: list[np.ndarray], n_frames: int) -> np.ndarray:
    kept = [a for a in arrs if a is not None and a.size]
    if not kept:
        return np.empty((n_frames, 0), dtype=np.float64)
    return np.hstack(kept)


def _get_atom(res, name: str):
    a = res.atoms.select_atoms(f"name {name}")
    return a[0] if len(a) == 1 else None


def _backbone_quads(protein_ag: "mda.core.groups.AtomGroup") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return backbone dihedral atom-index quads for phi/psi/omega.

    Definitions (standard):
      phi(i)   = C(i-1) - N(i)  - CA(i) - C(i)
      psi(i)   = N(i)   - CA(i) - C(i)  - N(i+1)
      omega(i) = CA(i)  - C(i)  - N(i+1)- CA(i+1)
    """
    residues = protein_ag.residues

    phi_quads: list[list[int]] = []
    psi_quads: list[list[int]] = []
    omg_quads: list[list[int]] = []

    for i, res in enumerate(residues):
        prev_res = residues[i - 1] if i > 0 else None
        next_res = residues[i + 1] if i + 1 < len(residues) else None

        n = _get_atom(res, "N")
        ca = _get_atom(res, "CA")
        c = _get_atom(res, "C")

        if prev_res is not None:
            c_prev = _get_atom(prev_res, "C")
            if c_prev is not None and n is not None and ca is not None and c is not None:
                phi_quads.append([c_prev.ix, n.ix, ca.ix, c.ix])

        if next_res is not None:
            n_next = _get_atom(next_res, "N")
            ca_next = _get_atom(next_res, "CA")
            if n is not None and ca is not None and c is not None and n_next is not None:
                psi_quads.append([n.ix, ca.ix, c.ix, n_next.ix])
            if ca is not None and c is not None and n_next is not None and ca_next is not None:
                omg_quads.append([ca.ix, c.ix, n_next.ix, ca_next.ix])

    return (
        np.asarray(phi_quads, dtype=np.int32),
        np.asarray(psi_quads, dtype=np.int32),
        np.asarray(omg_quads, dtype=np.int32),
    )


def _compute_dihedrals(
    u: mda.Universe, quads: np.ndarray, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute dihedrals (radians) for given quads across trajectory frames."""
    n_frames = len(u.trajectory[::stride])
    times = np.empty((n_frames,), dtype=np.float64)

    if quads.size == 0:
        for fi, ts in enumerate(u.trajectory[::stride]):
            times[fi] = ts.time
        return times, np.empty((n_frames, 0), dtype=np.float64)

    angles = np.empty((n_frames, quads.shape[0]), dtype=np.float64)
    for fi, ts in enumerate(u.trajectory[::stride]):
        times[fi] = ts.time
        pos = u.atoms.positions
        a = pos[quads[:, 0]]
        b = pos[quads[:, 1]]
        c = pos[quads[:, 2]]
        d = pos[quads[:, 3]]
        angles[fi, :] = calc_dihedrals(a, b, c, d)

    return times, angles


def main() -> None:
    p = argparse.ArgumentParser(
        description="Derive backbone dihedral features (phi/psi/omega) from a topology (.gro/.pdb) + .xtc"
    )
    p.add_argument(
        "-t",
        "--topology",
        required=True,
        help="Topology file (e.g. .gro or .pdb; must match trajectory atom count/order)",
    )
    p.add_argument("-x", "--trajectory", required=True, help="Trajectory file (.xtc)")
    p.add_argument(
        "-s",
        "--selection",
        default="protein",
        help="MDAnalysis selection for dihedrals (default: protein)",
    )
    p.add_argument("--stride", type=int, default=1, help="Frame stride (default: 1)")
    p.add_argument(
        "--out-prefix", default="dihedrals", help="Output prefix (default: dihedrals)"
    )
    args = p.parse_args()

    u = mda.Universe(args.topology, args.trajectory)
    ag = u.select_atoms(args.selection)
    if ag.n_atoms == 0:
        raise SystemExit(f"Selection matched 0 atoms: {args.selection!r}")

    phi_quads, psi_quads, omega_quads = _backbone_quads(ag)
    times_ps, phi = _compute_dihedrals(u, phi_quads, stride=args.stride)
    _, psi = _compute_dihedrals(u, psi_quads, stride=args.stride)
    _, omega = _compute_dihedrals(u, omega_quads, stride=args.stride)

    angles = _hstack_nonempty([phi, psi, omega], n_frames=times_ps.shape[0])
    if angles.size:
        features = np.hstack([np.sin(angles), np.cos(angles)])
    else:
        features = np.empty((angles.shape[0], 0), dtype=np.float64)

    np.savez_compressed(
        f"{args.out_prefix}_raw.npz",
        times_ps=times_ps,
        phi=phi,
        psi=psi,
        omega=omega,
        angles_stacked=angles,
        selection=args.selection,
        topology=args.topology,
        trajectory=args.trajectory,
        stride=args.stride,
        phi_quads=phi_quads,
        psi_quads=psi_quads,
        omega_quads=omega_quads,
    )
    np.save(f"{args.out_prefix}_features.npy", features)

    print(f"Wrote {args.out_prefix}_raw.npz")
    print(f"Wrote {args.out_prefix}_features.npy")
    print(
        f"Frames={features.shape[0]} dihedrals={angles.shape[1]} features={features.shape[1]}"
    )


if __name__ == "__main__":
    main()


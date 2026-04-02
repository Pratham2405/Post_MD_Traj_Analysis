#!/usr/bin/env python3
"""
TICA (Time-lagged Independent Component Analysis) on a saved feature matrix.

Input:
  - A .npy file containing X with shape (n_frames, n_features)

Outputs:
  - <prefix>_tica.npy: projected coordinates, shape (n_frames - lag, n_components)
  - <prefix>_tica_model.npz: TICA model parameters and metadata
  - <prefix>_tica.png (optional): IC1 vs IC2 scatter

Dependencies:
  - numpy
  - matplotlib (optional, for --plot)

Notes:
  - This is a single-trajectory TICA (no trajectory stitching / multiple runs).
  - Uses a numerically-stable whitening approach to solve:
        C_tau v = lambda C_0 v
"""

from __future__ import annotations

import argparse
import numpy as np


def _whiten_inverse_sqrt(C: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (C^{-1/2}, eigenvalues) for symmetric PSD C with diagonal regularization eps.
    """
    C = 0.5 * (C + C.T)
    C = C + eps * np.eye(C.shape[0], dtype=C.dtype)
    w, U = np.linalg.eigh(C)  # ascending
    w = np.maximum(w, eps)
    inv_sqrt = (U * (1.0 / np.sqrt(w))) @ U.T
    return inv_sqrt, w


def tica(
    X: np.ndarray,
    lag: int,
    n_components: int,
    *,
    zscore: bool,
    eps: float,
) -> dict[str, np.ndarray]:
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    n_frames, n_features = X.shape
    if lag < 1 or lag >= n_frames:
        raise ValueError(f"lag must be in [1, {n_frames-1}], got {lag}")

    if n_components < 1 or n_components > n_features:
        raise ValueError(f"n_components must be in [1, {n_features}], got {n_components}")

    # Center and (optionally) scale
    mean_ = X.mean(axis=0)
    Xc = X - mean_
    if zscore:
        scale_ = Xc.std(axis=0, ddof=1)
        scale_[scale_ == 0] = 1.0
        Xc = Xc / scale_
    else:
        scale_ = np.ones((n_features,), dtype=np.float64)

    X0 = Xc[:-lag, :]
    Xt = Xc[lag:, :]
    n = X0.shape[0]

    # Covariances
    C0 = (X0.T @ X0) / (n - 1)
    Ctau = (X0.T @ Xt) / (n - 1)
    Ctau = 0.5 * (Ctau + Ctau.T)  # enforce reversibility/symmetry (common in practice)

    # Solve generalized eigenproblem via whitening:
    #   Ctau v = lambda C0 v
    C0_inv_sqrt, C0_eigs = _whiten_inverse_sqrt(C0, eps=eps)
    K = C0_inv_sqrt @ Ctau @ C0_inv_sqrt
    K = 0.5 * (K + K.T)

    evals, evecs = np.linalg.eigh(K)  # ascending
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    evals_k = evals[:n_components]
    # Back-transform eigenvectors to original feature space
    components_ = (C0_inv_sqrt @ evecs[:, :n_components]).T  # (k, n_features)

    # Project: y_t = X0 @ components_.T
    coords = X0 @ components_.T  # (n_frames-lag, k)

    return {
        "coords": coords.astype(np.float64, copy=False),
        "components": components_.astype(np.float64, copy=False),
        "eigenvalues": evals_k.astype(np.float64, copy=False),
        "mean": mean_.astype(np.float64, copy=False),
        "scale": scale_.astype(np.float64, copy=False),
        "C0_eigs": C0_eigs.astype(np.float64, copy=False),
    }


def implied_timescales(evals: np.ndarray, lag: int, dt: float) -> np.ndarray:
    """
    Timescales in same time units as dt. For eigenvalues in (0,1):
      t_i = -lag*dt / ln(lambda_i)
    """
    out = np.full_like(evals, np.inf, dtype=np.float64)
    for i, lam in enumerate(evals):
        if lam > 0.0 and lam < 1.0:
            out[i] = -(lag * dt) / np.log(lam)
    return out


def plot_tica_2d(coords: np.ndarray, out_png: str, *, title: str | None, color_by_time: bool, dpi: int) -> None:
    if coords.shape[1] < 2:
        raise ValueError("Need at least 2 components to plot IC1 vs IC2.")
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: python3 -m pip install matplotlib"
        ) from e

    x = coords[:, 0]
    y = coords[:, 1]

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    if color_by_time:
        c = np.arange(coords.shape[0], dtype=np.float64)
        sc = ax.scatter(x, y, c=c, s=6, cmap="viridis", linewidths=0)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("frame index")
    else:
        ax.scatter(x, y, s=6, linewidths=0)

    ax.set_xlabel("IC1")
    ax.set_ylabel("IC2")
    ax.set_title(title or "TICA: IC1 vs IC2")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_tica_timeseries(
    coords: np.ndarray,
    out_png: str,
    *,
    title: str | None,
    dt: float,
    n_show: int,
    dpi: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: python3 -m pip install matplotlib"
        ) from e

    n_show = max(1, min(int(n_show), coords.shape[1]))
    t = np.arange(coords.shape[0], dtype=np.float64) * dt

    fig, ax = plt.subplots(figsize=(10.0, 4.5), constrained_layout=True)
    for i in range(n_show):
        ax.plot(t, coords[:, i], lw=0.8, label=f"IC{i+1}")
    ax.set_xlabel(f"time ({dt:g} units/frame)")
    ax.set_ylabel("IC value")
    ax.set_title(title or "TICA: IC time series")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=min(n_show, 5), fontsize=9, frameon=False)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_tica_density(
    coords: np.ndarray,
    out_png: str,
    *,
    title: str | None,
    bins: int,
    dpi: int,
) -> None:
    if coords.shape[1] < 2:
        raise ValueError("Need at least 2 components to plot IC1 vs IC2 density.")
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: python3 -m pip install matplotlib"
        ) from e

    x = coords[:, 0]
    y = coords[:, 1]
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    im = ax.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="magma",
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("counts")
    ax.set_xlabel("IC1")
    ax.set_ylabel("IC2")
    ax.set_title(title or "TICA: IC1 vs IC2 density")
    ax.grid(False)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_tica_fes(
    coords: np.ndarray,
    out_png: str,
    *,
    title: str | None,
    bins: int,
    kT: float,
    dpi: int,
) -> None:
    """
    Free energy surface estimate: F = -kT ln p(IC1,IC2), shifted to min=0.
    kT is in arbitrary energy units; set kT=1 for dimensionless F.
    """
    if coords.shape[1] < 2:
        raise ValueError("Need at least 2 components to plot IC1 vs IC2 FES.")
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: python3 -m pip install matplotlib"
        ) from e

    x = coords[:, 0]
    y = coords[:, 1]
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    # Avoid log(0)
    H = np.maximum(H, 1e-300)
    F = -kT * np.log(H)
    F = F - np.nanmin(F[np.isfinite(F)])

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    im = ax.imshow(
        F.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="viridis",
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("F (shifted)")
    ax.set_xlabel("IC1")
    ax.set_ylabel("IC2")
    ax.set_title(title or "TICA: IC1 vs IC2 free energy (FES)")
    ax.grid(False)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Run TICA on a saved feature matrix .npy")
    p.add_argument("-i", "--input", required=True, help="Input feature matrix (.npy), shape (frames, features)")
    p.add_argument("--lag", type=int, required=True, help="Lag in frames (integer, required)")
    p.add_argument("-k", "--n-components", type=int, default=5, help="Number of TICA components (default: 5)")
    p.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time step per frame (for implied timescales; default: 1.0 arbitrary units)",
    )
    p.add_argument(
        "--zscore",
        action="store_true",
        help="Z-score each feature before TICA (default: off). Often unnecessary for sin/cos dihedrals.",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Diagonal regularization for covariance (default: 1e-8)",
    )
    p.add_argument("--out-prefix", default="out", help="Output prefix (default: out)")
    p.add_argument("--plot", action="store_true", help="Save IC1 vs IC2 scatter plot to <prefix>_tica.png")
    p.add_argument("--no-time-color", action="store_true", help="Do not color plot by frame index")
    p.add_argument(
        "--plot-timeseries",
        action="store_true",
        help="Save IC time series plot to <prefix>_tica_timeseries.png",
    )
    p.add_argument(
        "--plot-density",
        action="store_true",
        help="Save IC1 vs IC2 2D histogram density to <prefix>_tica_density.png",
    )
    p.add_argument(
        "--plot-fes",
        action="store_true",
        help="Save IC1 vs IC2 free energy surface estimate to <prefix>_tica_fes.png",
    )
    p.add_argument("--bins", type=int, default=150, help="Bins for density/FES (default: 150)")
    p.add_argument("--kT", type=float, default=1.0, help="kT for FES scaling (default: 1.0)")
    p.add_argument("--n-show", type=int, default=3, help="Number of ICs to show in time series (default: 3)")
    args = p.parse_args()

    X = np.load(args.input)
    res = tica(X, lag=args.lag, n_components=args.n_components, zscore=args.zscore, eps=args.eps)

    coords = res["coords"]
    np.save(f"{args.out_prefix}_tica.npy", coords)

    its = implied_timescales(res["eigenvalues"], lag=args.lag, dt=args.dt)
    np.savez_compressed(
        f"{args.out_prefix}_tica_model.npz",
        components=res["components"],
        mean=res["mean"],
        scale=res["scale"],
        eigenvalues=res["eigenvalues"],
        implied_timescales=its,
        lag=int(args.lag),
        dt=float(args.dt),
        eps=float(args.eps),
        input=args.input,
        n_components=int(args.n_components),
        zscore=bool(args.zscore),
    )

    ev_str = ", ".join(f"{v:.6f}" for v in res["eigenvalues"][: min(5, res["eigenvalues"].size)])
    its_str = ", ".join(f"{t:.6g}" for t in its[: min(5, its.size)])
    print(f"Wrote {args.out_prefix}_tica.npy")
    print(f"Wrote {args.out_prefix}_tica_model.npz")
    print(f"Coords shape: {coords.shape}")
    print(f"Eigenvalues (first {min(5, res['eigenvalues'].size)}): {ev_str}")
    print(f"Implied timescales (first {min(5, its.size)}): {its_str} (time units)")

    if args.plot:
        out_png = f"{args.out_prefix}_tica.png"
        plot_tica_2d(
            coords,
            out_png,
            title=args.out_prefix,
            color_by_time=not args.no_time_color,
            dpi=200,
        )
        print(f"Wrote {out_png}")

    if args.plot_timeseries:
        out_png = f"{args.out_prefix}_tica_timeseries.png"
        plot_tica_timeseries(
            coords,
            out_png,
            title=args.out_prefix,
            dt=args.dt,
            n_show=args.n_show,
            dpi=200,
        )
        print(f"Wrote {out_png}")

    if args.plot_density:
        out_png = f"{args.out_prefix}_tica_density.png"
        plot_tica_density(
            coords,
            out_png,
            title=args.out_prefix,
            bins=args.bins,
            dpi=200,
        )
        print(f"Wrote {out_png}")

    if args.plot_fes:
        out_png = f"{args.out_prefix}_tica_fes.png"
        plot_tica_fes(
            coords,
            out_png,
            title=args.out_prefix,
            bins=args.bins,
            kT=args.kT,
            dpi=200,
        )
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()


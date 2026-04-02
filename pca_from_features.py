#!/usr/bin/env python3
"""
PCA on precomputed feature matrix (e.g., dihedral sin/cos features).

Input:
  - A .npy file containing X with shape (n_frames, n_features)

Outputs:
  - <prefix>_pca.npy: projected coordinates, shape (n_frames, n_components)
  - <prefix>_pca_model.npz: PCA model parameters and metadata

This implementation uses SVD (no scikit-learn required).
"""

from __future__ import annotations

import argparse
import numpy as np


def pca_svd(X: np.ndarray, n_components: int, zscore: bool) -> dict[str, np.ndarray]:
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    n_samples, n_features = X.shape
    if n_samples < 2:
        raise ValueError("Need at least 2 samples (frames) for PCA.")
    if n_components < 1 or n_components > min(n_samples, n_features):
        raise ValueError(
            f"n_components must be in [1, {min(n_samples, n_features)}], got {n_components}"
        )

    mean_ = X.mean(axis=0)
    Xc = X - mean_

    if zscore:
        scale_ = Xc.std(axis=0, ddof=1)
        scale_[scale_ == 0] = 1.0
        Xc = Xc / scale_
    else:
        scale_ = np.ones((n_features,), dtype=np.float64)

    # SVD on centered (and optionally scaled) data
    # Xc = U S Vt, principal axes are rows of Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    components_ = Vt[:n_components, :]  # (k, n_features)
    scores = Xc @ components_.T  # (n_samples, k)

    explained_variance_all = (S**2) / (n_samples - 1)  # (min(n_samples, n_features),)
    explained_variance_ = explained_variance_all[:n_components]
    total_var = explained_variance_all.sum()
    explained_variance_ratio_ = explained_variance_ / total_var if total_var > 0 else np.zeros_like(explained_variance_)

    return {
        "scores": scores.astype(np.float64, copy=False),
        "components": components_.astype(np.float64, copy=False),
        "mean": mean_.astype(np.float64, copy=False),
        "scale": scale_.astype(np.float64, copy=False),
        "explained_variance": explained_variance_.astype(np.float64, copy=False),
        "explained_variance_ratio": explained_variance_ratio_.astype(np.float64, copy=False),
    }


def plot_pca_2d(
    scores: np.ndarray,
    explained_variance_ratio: np.ndarray,
    out_png: str,
    *,
    title: str | None = None,
    color_by_time: bool = True,
    dpi: int = 200,
) -> None:
    if scores.shape[1] < 2:
        raise ValueError("Need at least 2 PCA components to plot PC1 vs PC2.")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: python3 -m pip install matplotlib"
        ) from e

    x = scores[:, 0]
    y = scores[:, 1]

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    if color_by_time:
        c = np.arange(scores.shape[0], dtype=np.float64)
        sc = ax.scatter(x, y, c=c, s=6, cmap="viridis", linewidths=0)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("frame index")
    else:
        ax.scatter(x, y, s=6, linewidths=0)

    evr1 = explained_variance_ratio[0] if explained_variance_ratio.size >= 1 else np.nan
    evr2 = explained_variance_ratio[1] if explained_variance_ratio.size >= 2 else np.nan
    ax.set_xlabel(f"PC1 ({evr1:.2%} var)")
    ax.set_ylabel(f"PC2 ({evr2:.2%} var)")
    ax.set_title(title or "PCA: PC1 vs PC2")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Run PCA (SVD) on a saved feature matrix .npy")
    p.add_argument("-i", "--input", required=True, help="Input feature matrix (.npy), shape (frames, features)")
    p.add_argument("-k", "--n-components", type=int, default=5, help="Number of PCA components (default: 5)")
    p.add_argument(
        "--zscore",
        action="store_true",
        help="Z-score each feature before PCA (default: off). Often unnecessary for sin/cos dihedrals.",
    )
    p.add_argument("--out-prefix", default="out", help="Output prefix (default: out)")
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save PC1 vs PC2 scatter plot to <prefix>_pca.png (default: off)",
    )
    p.add_argument(
        "--no-time-color",
        action="store_true",
        help="Do not color plot by frame index (default: colored by time)",
    )
    args = p.parse_args()

    X = np.load(args.input)
    res = pca_svd(X, n_components=args.n_components, zscore=args.zscore)

    np.save(f"{args.out_prefix}_pca.npy", res["scores"])
    np.savez_compressed(
        f"{args.out_prefix}_pca_model.npz",
        components=res["components"],
        mean=res["mean"],
        scale=res["scale"],
        explained_variance=res["explained_variance"],
        explained_variance_ratio=res["explained_variance_ratio"],
        input=args.input,
        n_components=args.n_components,
        zscore=bool(args.zscore),
    )

    evr = res["explained_variance_ratio"]
    evr_str = ", ".join(f"{v:.4f}" for v in evr[: min(5, len(evr))])
    print(f"Wrote {args.out_prefix}_pca.npy")
    print(f"Wrote {args.out_prefix}_pca_model.npz")
    print(f"Scores shape: {res['scores'].shape}")
    print(f"Explained variance ratio (first {min(5, len(evr))}): {evr_str}")

    if args.plot:
        out_png = f"{args.out_prefix}_pca.png"
        plot_pca_2d(
            res["scores"],
            res["explained_variance_ratio"],
            out_png,
            title=args.out_prefix,
            color_by_time=not args.no_time_color,
        )
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()


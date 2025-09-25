#!/usr/bin/env python3
"""
Analyze and visualize distribution of observations and actions in a buffer.

- Loads a single buffer file or multiple buffers (or a directory of .pt files)
- Extracts and flattens observations/actions from TensorDict buffers
- Generates per-dimension histograms and 2D PCA projections
- Computes summary stats and entropy-based concentration metrics
- Saves plots and a written summary report

Usage examples:
  python MLP/analyze_buffer_distribution.py --buffer_path path/to/buffer.pt
  python MLP/analyze_buffer_distribution.py --buffer_paths path/to/buffer_dir --max_samples 200000
  python MLP/analyze_buffer_distribution.py --buffer_path buffer.pt --out_dir analysis_out --bins 50 --dims_to_plot 12
"""

import argparse
import os
import glob
from typing import Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Buffer Loading Utilities
# -------------------------


def load_single_buffer(buffer_path: str, device: str = "cpu"):
    print(f"Loading single buffer: {buffer_path}")
    buffer_data = torch.load(buffer_path, weights_only=False, map_location=device)
    if hasattr(buffer_data, "shape") and hasattr(buffer_data, "keys"):
        print(f"Buffer shape: {buffer_data.shape}")
        print(f"Buffer keys: {list(buffer_data.keys())}")
        return buffer_data
    else:
        print(f"Buffer type: {type(buffer_data)}")
        return buffer_data


def load_multiple_buffers(buffer_paths, device: str = "cpu"):
    print("Loading multiple buffers...")

    if isinstance(buffer_paths, str):
        if os.path.isdir(buffer_paths):
            buffer_files = glob.glob(os.path.join(buffer_paths, "*.pt"))
            buffer_files = [f for f in buffer_files if "metadata" not in f]
            print(
                f"Found {len(buffer_files)} buffer files in directory: {buffer_paths}"
            )
        else:
            buffer_files = [buffer_paths]
    else:
        buffer_files = buffer_paths

    if not buffer_files:
        raise ValueError("No buffer files found!")

    print(f"Loading {len(buffer_files)} buffer files:")
    for i, file_path in enumerate(buffer_files):
        print(f"  {i + 1}. {file_path}")

    all_episodes = []
    for i, buffer_file in enumerate(buffer_files):
        print(f"\nLoading buffer {i + 1}/{len(buffer_files)}: {buffer_file}")
        try:
            buffer_data = torch.load(
                buffer_file, weights_only=False, map_location=device
            )
            if hasattr(buffer_data, "shape") and hasattr(buffer_data, "keys"):
                print(f"  Buffer shape: {buffer_data.shape}")
                print(f"  Buffer keys: {list(buffer_data.keys())}")
                all_episodes.append(buffer_data)
            else:
                print(f"  Buffer type (unsupported): {type(buffer_data)}")
        except Exception as e:
            print(f"  Error loading {buffer_file}: {e}")
            continue

    if not all_episodes:
        raise ValueError("No valid buffers were loaded!")

    print(f"\nCombining {len(all_episodes)} buffers...")
    try:
        if len(all_episodes) == 1:
            combined_buffer = all_episodes[0]
        else:
            from tensordict import TensorDict

            if all(isinstance(ep, TensorDict) for ep in all_episodes):
                combined_buffer = TensorDict.cat(all_episodes, dim=0)
            else:
                combined_buffer = torch.cat(all_episodes, dim=0)
        print(f"Combined buffer shape: {combined_buffer.shape}")
        print(f"Combined buffer keys: {list(combined_buffer.keys())}")
    except Exception as e:
        print(f"Error combining buffers: {e}")
        print("Using first buffer as fallback...")
        combined_buffer = all_episodes[0]

    return combined_buffer


# -------------------------
# Data Extraction
# -------------------------


def extract_obs_actions(
    buffer_data, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract observation and action tensors from a TensorDict buffer.
    Flattens episodic data into (N, dim) tensors.
    """
    if not (hasattr(buffer_data, "keys") and hasattr(buffer_data, "shape")):
        raise ValueError("Expected TensorDict-like buffer with .keys() and .shape")

    obs_key = "obs"
    act_key = "action"
    next_obs_key = "next_obs"
    terminated_key = "terminated"

    if obs_key is None or act_key is None:
        raise KeyError("Buffer must contain 'obs' and 'action' (or equivalent) keys.")

    obs = buffer_data[obs_key]
    actions = buffer_data[act_key]
    next_obs = buffer_data[next_obs_key]
    terminated = buffer_data[terminated_key]

    # Flatten episodic shapes: (E, T, D) -> (E*T, D)
    def flatten_if_needed(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.reshape(-1, x.shape[-1])
        if x.ndim == 2:
            return x
        raise ValueError(f"Unexpected tensor shape: {x.shape}")

    obs = flatten_if_needed(obs).float().to(device)
    # print(f"Obs shape: {obs.shape}")
    actions = flatten_if_needed(actions).float().to(device)
    # print(f"Actions shape: {actions.shape}")
    next_obs = flatten_if_needed(next_obs).float().to(device)
    # terminated = flatten_if_needed(terminated).float().to(device)
    if obs.shape[0] != actions.shape[0]:
        n = min(obs.shape[0], actions.shape[0])
        obs = obs[:n]
        actions = actions[:n]
        next_obs = next_obs[:n]
        terminated = terminated[:n]

    return obs, actions, next_obs, terminated


# -------------------------
# Metrics & Visualization
# -------------------------


def compute_summary_stats(x: np.ndarray):
    # Drop rows with any non-finite values to avoid NaNs propagating
    finite_rows = np.isfinite(x).all(axis=1)
    x_clean = x[finite_rows] if finite_rows.any() else np.zeros((0, x.shape[1]))
    return {
        "num_samples": int(x_clean.shape[0]),
        "dim": int(x_clean.shape[1]) if x_clean.size > 0 else int(x.shape[1]),
        "mean": (
            np.nanmean(x_clean, axis=0).tolist()
            if x_clean.size > 0
            else [float("nan")] * x.shape[1]
        ),
        "std": (
            np.nanstd(x_clean, axis=0).tolist()
            if x_clean.size > 0
            else [float("nan")] * x.shape[1]
        ),
        "min": (
            np.nanmin(x_clean, axis=0).tolist()
            if x_clean.size > 0
            else [float("nan")] * x.shape[1]
        ),
        "max": (
            np.nanmax(x_clean, axis=0).tolist()
            if x_clean.size > 0
            else [float("nan")] * x.shape[1]
        ),
    }


def normalized_entropy_per_dim(x: np.ndarray, num_bins: int = 30) -> np.ndarray:
    """
    Compute per-dimension entropy normalized to [0,1], where 1≈uniform and 0≈concentrated.
    Uses histogram bins over the observed range per dimension.
    """
    entropies = []
    # Drop any rows with non-finite values to ensure hist works
    finite_rows = np.isfinite(x).all(axis=1)
    X = x[finite_rows] if finite_rows.any() else np.zeros((0, x.shape[1]))
    for d in range(X.shape[1] if X.size > 0 else x.shape[1]):
        if X.size == 0:
            entropies.append(np.nan)
            continue
        col = X[:, d]
        # Avoid degenerate or NaN columns
        if not np.isfinite(col).any() or np.allclose(col, col[0]):
            entropies.append(0.0)
            continue
        # Guard against zero-variance with finite values
        col_min = np.nanmin(col)
        col_max = np.nanmax(col)
        if not np.isfinite(col_min) or not np.isfinite(col_max) or col_max == col_min:
            entropies.append(0.0)
            continue
        hist, _ = np.histogram(col, bins=num_bins, range=(col_min, col_max))
        p = hist.astype(np.float64)
        if p.sum() <= 0:
            entropies.append(0.0)
            continue
        p = p / p.sum()
        # Shannon entropy
        eps = 1e-12
        H = -(p * np.log(p + eps)).sum()
        nnz = max(1, int(np.count_nonzero(hist)))
        H_max = np.log(nnz)
        entropies.append(float(H / (H_max + eps)))
    return np.array(entropies)


def pca_2d(x: np.ndarray, standardize: bool = True) -> np.ndarray:
    """
    Simple PCA to 2D using SVD. Returns projected points (N, 2).
    """
    # Remove non-finite rows
    finite_rows = np.isfinite(x).all(axis=1)
    X = x[finite_rows] if finite_rows.any() else np.zeros((0, x.shape[1]))
    if X.shape[0] == 0 or X.shape[1] < 2:
        return X[:, :2] if X.shape[1] >= 2 else np.zeros((0, 2))
    if standardize:
        mu = np.nanmean(X, axis=0, keepdims=True)
        std = np.nanstd(X, axis=0, keepdims=True) + 1e-6
        X = (X - mu) / std
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    components = Vt[:2].T  # (D, 2)
    return X @ components


def plot_histograms(
    x: np.ndarray, title_prefix: str, out_dir: str, num_bins: int, dims_to_plot: int
):
    os.makedirs(out_dir, exist_ok=True)
    if x.size == 0:
        return
    # Remove non-finite rows per safety
    finite_rows = np.isfinite(x).all(axis=1)
    X = x[finite_rows] if finite_rows.any() else np.zeros((0, x.shape[1]))
    if X.shape[0] == 0:
        return
    dim = X.shape[1]
    k = min(dims_to_plot, dim)
    cols = 4
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        if i < k:
            col = X[:, i]
            if np.isfinite(col).any():
                finite_col = col[np.isfinite(col)]
                if finite_col.size > 0:
                    # Explicit range to avoid NaN edges
                    vmin, vmax = finite_col.min(), finite_col.max()
                    if vmin == vmax:
                        vmin, vmax = vmin - 1e-6, vmax + 1e-6
                    ax.hist(
                        finite_col,
                        bins=num_bins,
                        range=(vmin, vmax),
                        color="#4472C4",
                        alpha=0.85,
                    )
            ax.set_title(f"dim {i}")
        else:
            ax.axis("off")
    fig.suptitle(f"{title_prefix} histograms (first {k} dims)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(
        out_dir, f"{title_prefix.lower().replace(' ', '_')}_histograms.png"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pca(x: np.ndarray, title_prefix: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if x.shape[1] < 2:
        return
    z = pca_2d(x, standardize=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(z[:, 0], z[:, 1], gridsize=40, cmap="viridis")
    ax.set_title(f"{title_prefix} PCA (2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(hb, ax=ax, label="density")
    path = os.path.join(out_dir, f"{title_prefix.lower().replace(' ', '_')}_pca2d.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------
# CLI & Orchestration
# -------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze observation and action distributions from buffer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--buffer_path", type=str, help="Path to a single buffer .pt file"
    )
    group.add_argument(
        "--buffer_paths",
        type=str,
        nargs="+",
        help="Paths to multiple buffer .pt files or directory",
    )

    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="buffer_analysis",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200000,
        help="Subsample for faster analysis (0=all)",
    )
    parser.add_argument(
        "--bins", type=int, default=30, help="Histogram bins per dimension"
    )
    parser.add_argument(
        "--dims_to_plot",
        type=int,
        default=16,
        help="Number of dims to plot in histograms",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )

    # Load buffer(s)
    if args.buffer_path:
        buffer_data = load_single_buffer(args.buffer_path, device=str(device))
    else:
        if len(args.buffer_paths) == 1:
            buffer_data = load_multiple_buffers(
                args.buffer_paths[0], device=str(device)
            )
        else:
            buffer_data = load_multiple_buffers(args.buffer_paths, device=str(device))

    # Extract obs/actions and move to CPU numpy
    obs, actions, next_obs, terminated = extract_obs_actions(buffer_data, device=str(device))
    print(f"Obs shape: {obs.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Next obs shape: {next_obs.shape}")
    print(f"Terminated shape: {terminated.shape}")
    print(f"First few elements in observations: {obs[:5]}")
    print(f"First few elements in actions: {actions[:5]}")

    # Subsample if requested
    N = obs.shape[0]
    if args.max_samples and args.max_samples > 0 and N > args.max_samples:
        idx = torch.randperm(N)[: args.max_samples]
        obs = obs[idx]
        actions = actions[idx]
        N = obs.shape[0]
        print(f"Subsampled to {N} samples for analysis.")

    obs_np = obs.cpu().numpy()
    act_np = actions.cpu().numpy()

    os.makedirs(args.out_dir, exist_ok=True)

    # Summary stats
    obs_stats = compute_summary_stats(obs_np)
    act_stats = compute_summary_stats(act_np)

    # Entropy-based concentration (per-dimension, then summarize)
    obs_ent = normalized_entropy_per_dim(obs_np, num_bins=args.bins)
    act_ent = normalized_entropy_per_dim(act_np, num_bins=args.bins)

    obs_ent_mean = float(np.nanmean(obs_ent))
    act_ent_mean = float(np.nanmean(act_ent))

    def verdict(ent_mean: float) -> str:
        if ent_mean >= 0.85:
            return "largely uniform/covering"
        if ent_mean >= 0.65:
            return "moderately diverse"
        return "concentrated on fewer modes"

    # Save summary report
    report_path = os.path.join(args.out_dir, "summary.txt")
    with open(report_path, "w") as f:
        f.write("OBSERVATIONS\n")
        f.write(f"  samples: {obs_stats['num_samples']}  dim: {obs_stats['dim']}\n")
        f.write(f"  mean[:5]: {np.array(obs_stats['mean'])[:5]}\n")
        f.write(f"  std[:5]:  {np.array(obs_stats['std'])[:5]}\n")
        f.write(f"  min[:5]:  {np.array(obs_stats['min'])[:5]}\n")
        f.write(f"  max[:5]:  {np.array(obs_stats['max'])[:5]}\n")
        f.write(
            f"  normalized entropy (mean over dims): {obs_ent_mean:.3f} -> {verdict(obs_ent_mean)}\n\n"
        )

        f.write("ACTIONS\n")
        f.write(f"  samples: {act_stats['num_samples']}  dim: {act_stats['dim']}\n")
        f.write(f"  mean[:5]: {np.array(act_stats['mean'])[:5]}\n")
        f.write(f"  std[:5]:  {np.array(act_stats['std'])[:5]}\n")
        f.write(f"  min[:5]:  {np.array(act_stats['min'])[:5]}\n")
        f.write(f"  max[:5]:  {np.array(act_stats['max'])[:5]}\n")
        f.write(
            f"  normalized entropy (mean over dims): {act_ent_mean:.3f} -> {verdict(act_ent_mean)}\n"
        )

    print(f"Saved summary to {report_path}")

    # Plots
    plot_histograms(
        obs_np,
        "Observations",
        args.out_dir,
        num_bins=args.bins,
        dims_to_plot=args.dims_to_plot,
    )
    plot_histograms(
        act_np,
        "Actions",
        args.out_dir,
        num_bins=args.bins,
        dims_to_plot=min(args.dims_to_plot, act_np.shape[1]),
    )

    plot_pca(obs_np, "Observations", args.out_dir)
    plot_pca(act_np, "Actions", args.out_dir)

    print("Visualization saved:")
    print(f"  - {os.path.join(args.out_dir, 'observations_histograms.png')}")
    print(f"  - {os.path.join(args.out_dir, 'actions_histograms.png')}")
    print(f"  - {os.path.join(args.out_dir, 'observations_pca2d.png')}")
    print(f"  - {os.path.join(args.out_dir, 'actions_pca2d.png')}")


if __name__ == "__main__":
    main()

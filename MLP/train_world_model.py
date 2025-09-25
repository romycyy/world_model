#!/usr/bin/env python3
"""
Train a vanilla MLP world model that predicts next observation from (observation, action),
using the same buffer data format used elsewhere in this repo.

- Supports loading a single .pt buffer or multiple buffers (files or a directory)
- Handles episodic TensorDict buffers by aligning (obs_t, action_t) -> obs_{t+1}
- Handles flattened buffers if explicit next observation keys are present
- Configurable MLP depth/width, dropout, and delta prediction
"""

# How this script works:
# 1) Load buffer file(s) saved by TD-MPC2 (TensorDict format).
# 2) Convert them into aligned (obs_t, action_t) -> next_obs pairs.
# 3) Build DataLoaders and train WorldModelMLP with MSE loss to next_obs.
# 4) Save a checkpoint including model hyperparameters and training curves.

import argparse
import os
import glob
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Try imports regardless of execution location
try:
    from MLP.world_model import WorldModelMLP
except Exception:
    from world_model import WorldModelMLP


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
    """
    Load and combine multiple buffer files.

    Args:
        buffer_paths: List of paths to buffer files or directory containing buffers
        device: Device to move data to

    Returns:
        combined_buffer_data: Combined TensorDict from all buffers
    """
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
# Data Processing
# -------------------------


def process_buffer_for_world_model(
    buffer_data,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a TensorDict-like buffer into flattened (obs_t, action_t, next_obs_t) tensors.

    Expects keys: 'obs', 'action', 'next_obs'. Shapes may be episodic (E, T, D) or flat (N, D).
    If 'next_obs' is missing but episodic shapes are present, it will be derived by shifting 'obs'.
    Returns float32 tensors on the requested device with matched lengths.
    """
    if not (hasattr(buffer_data, "keys") and hasattr(buffer_data, "shape")):
        raise ValueError("Expected TensorDict-like buffer with .keys() and .shape")

    obs_key = "obs"
    act_key = "action"
    next_obs_key = "next_obs"

    keys = set(buffer_data.keys())
    if obs_key not in keys or act_key not in keys:
        raise KeyError("Buffer must contain 'obs' and 'action' keys.")

    obs = buffer_data[obs_key]
    actions = buffer_data[act_key]

    # Derive next_obs if missing and episodic shapes are present
    if next_obs_key in keys:
        next_obs = buffer_data[next_obs_key]
    else:
        if obs.ndim == 3 and actions.ndim == 3 and obs.shape[:2] == actions.shape[:2]:
            # Align (obs_t, action_t) -> obs_{t+1} within each episode
            if obs.shape[1] < 2:
                raise ValueError("Cannot derive next_obs: sequence length < 2.")
            next_obs = obs[:, 1:, :]
            obs = obs[:, :-1, :]
            actions = actions[:, :-1, :]
        else:
            raise KeyError(
                "Buffer missing 'next_obs' and cannot derive from non-episodic shapes."
            )

    def flatten_if_needed(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.reshape(-1, x.shape[-1])
        if x.ndim == 2:
            return x
        raise ValueError(f"Unexpected tensor shape: {x.shape}")

    obs = flatten_if_needed(obs).float()
    actions = flatten_if_needed(actions).float()
    next_obs = flatten_if_needed(next_obs).float()

    # Trim to min length to ensure alignment
    n = min(obs.shape[0], actions.shape[0], next_obs.shape[0])
    obs = obs[:n]
    actions = actions[:n]
    next_obs = next_obs[:n]

    # Filter out any rows with non-finite values for stability
    finite_mask = (
        torch.isfinite(obs).all(dim=1)
        & torch.isfinite(actions).all(dim=1)
        & torch.isfinite(next_obs).all(dim=1)
    )
    if finite_mask.sum().item() != n:
        kept = int(finite_mask.sum().item())
        dropped = int(n - kept)
        print(f"Filtered out {dropped} non-finite samples; keeping {kept}.")
        obs = obs[finite_mask]
        actions = actions[finite_mask]
        next_obs = next_obs[finite_mask]

    obs = obs.to(device)
    actions = actions.to(device)
    next_obs = next_obs.to(device)

    return obs, actions, next_obs


# -------------------------
# DataLoader Utilities
# -------------------------


def create_dataloaders(
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    batch_size: int = 256,
    train_split: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    dataset = TensorDataset(obs, actions, next_obs)
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader


# -------------------------
# Metrics & Evaluation
# -------------------------


def compute_batch_accuracy(
    pred: torch.Tensor, target: torch.Tensor, tol: float
) -> float:
    """
    Define accuracy for regression as the fraction of samples whose mean absolute
    error per-dimension is below a tolerance `tol`.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {pred.shape} vs {target.shape}"
        )
    mae_per_sample = (pred - target).abs().mean(dim=-1)
    correct = (mae_per_sample <= tol).float()
    return float(correct.mean().item())


def evaluate_loader(
    model: nn.Module, loader: DataLoader, device: str, tol: float
) -> Tuple[float, float]:
    """Return (avg_mse, accuracy_within_tol) over the given loader."""
    model.eval()
    criterion = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    with torch.no_grad():
        for obs_batch, act_batch, next_obs_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            next_obs_batch = next_obs_batch.to(device)
            pred_next = model(obs_batch, act_batch)
            loss = criterion(pred_next, next_obs_batch)
            acc = compute_batch_accuracy(pred_next, next_obs_batch, tol)
            total_loss += float(loss.item())
            total_acc += float(acc)
            batches += 1
    avg_mse = total_loss / max(batches, 1)
    avg_acc = total_acc / max(batches, 1)
    return avg_mse, avg_acc


# -------------------------
# Training Loop
# -------------------------


def train_world_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    acc_tol: float = 0.05,
) -> Tuple[List[float], List[float]]:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses: List[float] = []
    val_losses: List[float] = []

    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Standard supervised learning: predict next_obs and minimize MSE
    for epoch in range(num_epochs):
        # Train
        model.train()
        running_loss = 0.0
        batches = 0

        for obs_batch, act_batch, next_obs_batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]", leave=False
        ):
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            next_obs_batch = next_obs_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_next = model(obs_batch, act_batch)
            loss = criterion(pred_next, next_obs_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        avg_train = running_loss / max(batches, 1)
        train_losses.append(avg_train)

        # Validate
        model.eval()
        val_running = 0.0
        val_batches = 0
        with torch.no_grad():
            for obs_batch, act_batch, next_obs_batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [val]", leave=False
            ):
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                next_obs_batch = next_obs_batch.to(device)
                pred_next = model(obs_batch, act_batch)
                val_loss = criterion(pred_next, next_obs_batch)
                val_running += val_loss.item()
                val_batches += 1

        avg_val = val_running / max(val_batches, 1)
        val_losses.append(avg_val)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}: Train Loss = {avg_train:.6f}, Val Loss = {avg_val:.6f}"
            )

    # End-of-training evaluation: MSE and accuracy
    tolerances = [0.005, 0.01, 0.015, 0.02]
    for tol in tolerances:
        train_mse, train_acc = evaluate_loader(model, train_loader, device, tol)
        val_mse, val_acc = evaluate_loader(model, val_loader, device, tol)
        print(
            f"Tolerance {tol}: Train MSE={train_mse:.6f}, Acc@tol({tol})={train_acc:.4f} | "
            f"Val MSE={val_mse:.6f}, Acc@tol({tol})={val_acc:.4f}"
        )

    return train_losses, val_losses


# -------------------------
# CLI
# -------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train an MLP world model (obs, action -> next_obs) from buffer data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on a single buffer file
  python MLP/train_world_model.py --buffer_path path/to/buffer.pt

  # Train on all buffers in a directory
  python MLP/train_world_model.py --buffer_paths path/to/buffer_dir

  # Train with delta prediction and custom MLP
  python MLP/train_world_model.py --buffer_path buffer.pt --predict_delta --hidden_dims 512 512 256 --dropout 0.1
        """,
    )

    buffer_group = parser.add_mutually_exclusive_group(required=False)
    buffer_group.add_argument(
        "--buffer_path", type=str, help="Path to a single buffer .pt file"
    )
    buffer_group.add_argument(
        "--buffer_paths",
        type=str,
        nargs="+",
        help="Paths to multiple buffer .pt files or directory",
    )

    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer dimensions",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument(
        "--predict_delta",
        action="store_true",
        help="Predict state delta instead of absolute next state",
    )

    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )

    parser.add_argument(
        "--save_model",
        type=str,
        default="world_model.pt",
        help="Path to save the trained model",
    )

    parser.add_argument(
        "--acc_tol",
        type=float,
        default=0.05,
        help="Tolerance used to compute accuracy (fraction of samples with MAE per-dim <= tol)",
    )

    args = parser.parse_args()

    if not args.buffer_path and not args.buffer_paths:
        parser.error("Either --buffer_path or --buffer_paths is required for training")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load buffer
    if args.buffer_path:
        buffer_data = load_single_buffer(args.buffer_path, device=str(device))
    else:
        # If a single string provided in nargs+, it may be a directory
        if len(args.buffer_paths) == 1:
            buffer_data = load_multiple_buffers(
                args.buffer_paths[0], device=str(device)
            )
        else:
            buffer_data = load_multiple_buffers(args.buffer_paths, device=str(device))

    # Process into (obs, action, next_obs)
    obs_tensor, action_tensor, next_obs_tensor = process_buffer_for_world_model(
        buffer_data, device=str(device)
    )

    # Build model
    obs_dim = obs_tensor.shape[1]
    action_dim = action_tensor.shape[1]
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")

    model = WorldModelMLP(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        predict_delta=args.predict_delta,
    )

    # Dataloaders
    train_loader, val_loader = create_dataloaders(
        obs_tensor, action_tensor, next_obs_tensor, batch_size=args.batch_size
    )

    # Peek a few samples to verify inputs
    def print_sample_from_loader(name: str, loader: DataLoader):
        try:
            batch = next(iter(loader))
        except StopIteration:
            print(f"{name}: loader is empty")
            return
        obs_b, act_b, next_b = batch
        print(
            f"{name} sample batch shapes: obs={tuple(obs_b.shape)}, act={tuple(act_b.shape)}, next={tuple(next_b.shape)}"
        )
        k = min(2, obs_b.shape[0])
        print(f"{name} first {k} obs rows (cpu):\n{obs_b[:k].detach().to('cpu')}")
        print(f"{name} first {k} act rows (cpu):\n{act_b[:k].detach().to('cpu')}")
        print(f"{name} first {k} next rows (cpu):\n{next_b[:k].detach().to('cpu')}")

    print_sample_from_loader("Train", train_loader)
    print_sample_from_loader("Val", val_loader)

    # Train
    print("Starting world model training...")
    train_losses, val_losses = train_world_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=str(device),
        acc_tol=float(args.acc_tol),
    )

    # Save checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "predict_delta": args.predict_delta,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    if args.buffer_path:
        ckpt["buffer_path"] = args.buffer_path
    else:
        ckpt["buffer_paths"] = args.buffer_paths

    torch.save(ckpt, args.save_model)
    print(f"Model saved to: {args.save_model}")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Val Loss: {val_losses[-1]:.6f}")


if __name__ == "__main__":
    main()

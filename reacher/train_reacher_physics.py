#!/usr/bin/env python3
"""
Train the physics-informed world model for the reacher environment.

This script trains a symbolic physics model for a 2-link planar reacher
using analytical dynamics equations derived from Lagrangian mechanics.

Key features:
- Learnable physical parameters (masses, lengths, inertias, damping)
- Parameter constraints for physical validity
- Analytical mass matrix computation
- RK4 integration option for accuracy
- Gradient clipping for stability
- Learning rate scheduling
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add workspace root to Python path
workspace_root = Path(__file__).resolve().parents[1]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from reacher.physical_model_reacher import ReacherPhysicsModel  # noqa: E402
from MLP.utils import (  # noqa: E402
    load_single_buffer,
    load_multiple_buffers,
    process_buffer_for_world_model,
)


def create_dataloaders(
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    batch_size: int = 256,
    train_split: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
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


def compute_batch_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    tol: float,
    normalization_std: torch.Tensor = None,
) -> float:
    """
    Compute accuracy as fraction of samples with mean normalized error <= tol.

    Normalized error = |pred - target| / std_dev(dimension)
    This makes errors comparable across dimensions with different units/scales.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {pred.shape} vs {target.shape}"
        )

    abs_diff_per_element = torch.abs(pred - target)

    if normalization_std is not None:
        # Ensure normalization_std is on the same device as predictions
        if normalization_std.device != pred.device:
            normalization_std = normalization_std.to(pred.device)
        # Use robust normalization: for dimensions with very low variance,
        # use absolute error instead of normalized error to avoid division by near-zero
        # Threshold: if std < 0.01, consider the dimension as near-constant
        min_std_threshold = 0.01
        robust_std = torch.maximum(
            normalization_std,
            torch.tensor(min_std_threshold, device=normalization_std.device),
        )
        normalized_diff = abs_diff_per_element / robust_std
    else:
        normalized_diff = abs_diff_per_element

    mae_per_sample = normalized_diff.mean(dim=-1)
    correct = (mae_per_sample <= tol).float()
    return float(correct.mean().item())


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    tol: float,
    normalization_std: torch.Tensor = None,
) -> Tuple[float, float]:
    """Evaluate model on a data loader."""
    model.eval()
    criterion = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for obs_batch, act_batch, next_obs_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            next_obs_batch = next_obs_batch.to(device)

            pred_next = model(obs_batch, act_batch)
            loss = criterion(pred_next, next_obs_batch)
            acc = compute_batch_accuracy(
                pred_next, next_obs_batch, tol, normalization_std
            )

            total_loss += float(loss.item())
            total_acc += float(acc)
            num_batches += 1

    avg_mse = total_loss / max(num_batches, 1)
    avg_acc = total_acc / max(num_batches, 1)
    return avg_mse, avg_acc


def train_reacher_physics(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    normalization_std: torch.Tensor,
    num_epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    acc_tol: float = 0.05,
    target_weight: float = 0.1,
    scheduler_step_size: int = 5,
    scheduler_gamma: float = 0.5,
    grad_clip: float = 1.0,
) -> Tuple[List[float], List[float]]:
    """
    Train the physics-based reacher model.

    Args:
        model: Reacher physics model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        normalization_std: Standard deviation per dimension for normalized error metrics
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        acc_tol: Tolerance for accuracy computation
        target_weight: Weight for target position loss component (should be low since target is constant)
        scheduler_step_size: Number of epochs between LR reductions
        scheduler_gamma: Factor by which to reduce LR
        grad_clip: Gradient clipping threshold (0 to disable)

    Returns:
        Tuple of (train_losses, val_losses) for each epoch
    """
    model = model.to(device)

    train_losses: List[float] = []
    val_losses: List[float] = []

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
        verbose=True,
    )

    # Print configuration
    num_params = sum(p.numel() for p in model.parameters())

    print("\nTraining Configuration:")
    print(f"  Device: {device}")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Target weight: {target_weight}")
    print(f"  Gradient clipping: {grad_clip}")
    print(
        f"  LR Scheduler: StepLR (step_size={scheduler_step_size}, gamma={scheduler_gamma})"
    )

    # Print initial parameter values
    if hasattr(model, "print_learned_parameters"):
        print("\nInitial Physical Parameters:")
        model.print_learned_parameters()

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        num_batches = 0

        for obs_batch, act_batch, next_obs_batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]", leave=False
        ):
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            next_obs_batch = next_obs_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss = model.compute_loss(
                observation=obs_batch,
                action=act_batch,
                target_next_observation=next_obs_batch,
                target_weight=target_weight,
            )

            loss.backward()

            # Gradient clipping for stability
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        avg_train_loss = running_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_num_batches = 0

        with torch.no_grad():
            for obs_batch, act_batch, next_obs_batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [val]", leave=False
            ):
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                next_obs_batch = next_obs_batch.to(device)

                val_loss = model.compute_loss(
                    observation=obs_batch,
                    action=act_batch,
                    target_next_observation=next_obs_batch,
                    target_weight=target_weight,
                )
                val_running_loss += float(val_loss.item())
                val_num_batches += 1

        avg_val_loss = val_running_loss / max(val_num_batches, 1)
        val_losses.append(avg_val_loss)

        # Step the scheduler
        scheduler.step()

        # Periodic logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}, LR = {current_lr:.2e}"
            )

    # End-of-training evaluation with multiple tolerances

    # First, let's check what the actual normalized errors look like
    print("\nDiagnosing normalized errors...")
    model.eval()
    with torch.no_grad():
        # Get a sample batch to diagnose
        sample_obs, sample_act, sample_next = next(iter(val_loader))
        sample_obs = sample_obs.to(device)
        sample_act = sample_act.to(device)
        sample_next = sample_next.to(device)
        sample_pred = model(sample_obs, sample_act)

        abs_err = torch.abs(sample_pred - sample_next)
        norm_std_device = normalization_std.to(device)

        print(f"Sample batch size: {sample_obs.shape[0]}")
        print(
            f"Normalization std device: {norm_std_device.device}, shape: {norm_std_device.shape}"
        )
        print(
            f"Normalization std stats: min={norm_std_device.min().item():.6f}, max={norm_std_device.max().item():.6f}, mean={norm_std_device.mean().item():.6f}"
        )

        # Identify near-constant dimensions
        min_std_threshold = 0.01
        low_variance_dims = (norm_std_device < min_std_threshold).nonzero(
            as_tuple=True
        )[0]
        if len(low_variance_dims) > 0:
            print(
                f"\nWARNING: {len(low_variance_dims)} dimensions have very low variance (std < {min_std_threshold}):"
            )
            dim_names = [
                "cos(θ1)",
                "cos(θ2)",
                "sin(θ1)",
                "sin(θ2)",
                "target_x",
                "target_y",
                "θ1_dot",
                "θ2_dot",
                "to_target_x",
                "to_target_y",
            ]
            for dim_idx in low_variance_dims[:10]:  # Show first 10
                dim_name = (
                    dim_names[dim_idx.item()]
                    if dim_idx.item() < len(dim_names)
                    else f"Dim {dim_idx.item()}"
                )
                print(f"  {dim_name}: std={norm_std_device[dim_idx].item():.8f}")
            if len(low_variance_dims) > 10:
                print(f"  ... and {len(low_variance_dims) - 10} more")
            print(
                f"These dimensions will use absolute error with threshold={min_std_threshold} for normalization."
            )

        # Use robust normalization (same as in compute_batch_accuracy)
        robust_std = torch.maximum(
            norm_std_device, torch.tensor(min_std_threshold, device=device)
        )
        normalized_err = abs_err / robust_std
        mean_norm_err_per_sample = normalized_err.mean(dim=-1)

        print(
            f"\nAbsolute error stats: min={abs_err.min().item():.6f}, max={abs_err.max().item():.6f}, mean={abs_err.mean().item():.6f}"
        )
        print(
            f"Robust normalized error stats: min={normalized_err.min().item():.6f}, max={normalized_err.max().item():.6f}, mean={normalized_err.mean().item():.6f}"
        )
        print(
            f"Mean normalized error per sample stats: min={mean_norm_err_per_sample.min().item():.6f}, max={mean_norm_err_per_sample.max().item():.6f}, mean={mean_norm_err_per_sample.mean().item():.6f}"
        )
        print(
            f"Number of samples with mean_norm_err <= 1.0: {(mean_norm_err_per_sample <= 1.0).sum().item()} / {len(mean_norm_err_per_sample)}"
        )
        print(
            f"Number of samples with mean_norm_err <= 3.0: {(mean_norm_err_per_sample <= 3.0).sum().item()} / {len(mean_norm_err_per_sample)}"
        )

    print("\nFinal Evaluation:")
    print("(Tolerance in units of standard deviations per dimension)")
    # Tolerances are in units of standard deviations (normalized error)
    tolerances = [0.5, 1.0, 1.5, 2.0, 3.0]
    for tol in tolerances:
        train_mse, train_acc = evaluate_loader(
            model, train_loader, device, tol, normalization_std
        )
        val_mse, val_acc = evaluate_loader(
            model, val_loader, device, tol, normalization_std
        )
        print(
            f"  Tolerance {tol:.1f}σ: Train acc={train_acc:.4f}, Val acc={val_acc:.4f}"
        )

    # Print learned parameters
    if hasattr(model, "print_learned_parameters"):
        print("\nFinal Learned Physical Parameters:")
        model.print_learned_parameters()

    return train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(
        description="Train a physics-informed world model for the reacher environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on a single buffer file with default settings
  python reacher/train_reacher_physics.py --buffer_path path/to/buffer.pt

  # Train with RK4 integration (recommended)
  python reacher/train_reacher_physics.py --buffer_path buffer.pt --use_rk4

  # Train with Euler integration (faster but less accurate)
  python reacher/train_reacher_physics.py --buffer_path buffer.pt --no_rk4

  # Train with custom hyperparameters
  python reacher/train_reacher_physics.py --buffer_path buffer.pt --epochs 200 --lr 5e-4 --batch_size 512
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
        "--device", type=str, default="cuda", help="Device (auto/cpu/cuda)"
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="reacher_physics_model.pt",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--acc_tol",
        type=float,
        default=0.05,
        help="Tolerance for accuracy computation",
    )

    # Physics parameters
    parser.add_argument(
        "--dt", type=float, default=0.02, help="Integration timestep (s)"
    )
    parser.add_argument(
        "--mass_link1", type=float, default=0.1, help="Mass of link 1 (kg)"
    )
    parser.add_argument(
        "--mass_link2", type=float, default=0.1, help="Mass of link 2 (kg)"
    )
    parser.add_argument(
        "--length_link1", type=float, default=0.1, help="Length of link 1 (m)"
    )
    parser.add_argument(
        "--length_link2", type=float, default=0.1, help="Length of link 2 (m)"
    )
    parser.add_argument(
        "--com_link1", type=float, default=0.05, help="Center of mass of link 1 (m)"
    )
    parser.add_argument(
        "--com_link2", type=float, default=0.05, help="Center of mass of link 2 (m)"
    )
    parser.add_argument(
        "--inertia_link1", type=float, default=0.001, help="Inertia of link 1 (kg⋅m²)"
    )
    parser.add_argument(
        "--inertia_link2", type=float, default=0.001, help="Inertia of link 2 (kg⋅m²)"
    )
    parser.add_argument(
        "--joint1_damping",
        type=float,
        default=0.01,
        help="Damping coefficient for joint 1",
    )
    parser.add_argument(
        "--joint2_damping",
        type=float,
        default=0.01,
        help="Damping coefficient for joint 2",
    )
    parser.add_argument(
        "--torque_scale_1",
        type=float,
        default=1.0,
        help="Torque scaling factor for joint 1",
    )
    parser.add_argument(
        "--torque_scale_2",
        type=float,
        default=1.0,
        help="Torque scaling factor for joint 2",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=0.0,
        help="Gravity component (m/s², usually 0 for horizontal)",
    )
    parser.add_argument(
        "--target_weight",
        type=float,
        default=0.1,
        help="Weight for target position prediction in loss (should be low since target is constant)",
    )

    # Model improvements
    parser.add_argument(
        "--use_rk4",
        action="store_true",
        default=True,
        help="Use RK4 integration (default: True)",
    )
    parser.add_argument(
        "--no_rk4",
        action="store_false",
        dest="use_rk4",
        help="Use semi-implicit Euler instead of RK4",
    )
    parser.add_argument(
        "--mass_matrix_reg",
        type=float,
        default=1e-6,
        help="Regularization for mass matrix",
    )

    # Training hyperparameters
    parser.add_argument(
        "--scheduler_step_size",
        type=int,
        default=5,
        help="Number of epochs between LR reductions",
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.5,
        help="Factor to reduce LR",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping threshold (0 to disable)",
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

    if obs_dim < 10:
        raise ValueError(
            f"ReacherPhysicsModel expects obs_dim >= 10, got obs_dim={obs_dim}"
        )
    if action_dim != 2:
        raise ValueError(
            f"ReacherPhysicsModel expects action_dim=2, got action_dim={action_dim}"
        )

    if obs_dim > 10:
        print(
            f"Note: obs_dim={obs_dim} > 10. Extra dimensions (indices 10+) will be passed through unchanged."
        )

    model = ReacherPhysicsModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        dt=float(args.dt),
        mass_link1=float(args.mass_link1),
        mass_link2=float(args.mass_link2),
        length_link1=float(args.length_link1),
        length_link2=float(args.length_link2),
        com_link1=float(args.com_link1),
        com_link2=float(args.com_link2),
        inertia_link1=float(args.inertia_link1),
        inertia_link2=float(args.inertia_link2),
        joint1_damping=float(args.joint1_damping),
        joint2_damping=float(args.joint2_damping),
        torque_scale_1=float(args.torque_scale_1),
        torque_scale_2=float(args.torque_scale_2),
        gravity=float(args.gravity),
        use_rk4=args.use_rk4,
        mass_matrix_reg=args.mass_matrix_reg,
    )

    print("\nModel Configuration:")
    print(f"  Use RK4 integration: {args.use_rk4}")
    print(f"  Mass matrix regularization: {args.mass_matrix_reg}")

    # Dataloaders
    train_loader, val_loader = create_dataloaders(
        obs_tensor, action_tensor, next_obs_tensor, batch_size=args.batch_size
    )

    # Compute normalization statistics for fair comparison across dimensions
    normalization_std = next_obs_tensor.std(dim=0).to(device)
    print("\nNormalization std per dimension:")
    dim_names = [
        "cos(θ1)",
        "cos(θ2)",
        "sin(θ1)",
        "sin(θ2)",
        "target_x",
        "target_y",
        "θ1_dot",
        "θ2_dot",
        "to_target_x",
        "to_target_y",
    ]
    for i, (std_val, name) in enumerate(
        zip(normalization_std.cpu().numpy(), dim_names)
    ):
        print(f"  Dim {i} ({name}): {std_val:.6f}")

    # Train
    print("\n" + "=" * 60)
    print("Starting Reacher Physics Model Training")
    print("=" * 60)
    train_losses, val_losses = train_reacher_physics(
        model,
        train_loader,
        val_loader,
        normalization_std,
        num_epochs=args.epochs,
        lr=args.lr,
        device=str(device),
        acc_tol=float(args.acc_tol),
        target_weight=float(args.target_weight),
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        grad_clip=args.grad_clip,
    )

    # Save checkpoint
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)

    learned_params = {}
    if hasattr(model, "get_learned_parameters"):
        learned_params = model.get_learned_parameters()

    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_class": "ReacherPhysicsModel",
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "initial_params": {
            "dt": float(args.dt),
            "mass_link1": float(args.mass_link1),
            "mass_link2": float(args.mass_link2),
            "length_link1": float(args.length_link1),
            "length_link2": float(args.length_link2),
            "com_link1": float(args.com_link1),
            "com_link2": float(args.com_link2),
            "inertia_link1": float(args.inertia_link1),
            "inertia_link2": float(args.inertia_link2),
            "joint1_damping": float(args.joint1_damping),
            "joint2_damping": float(args.joint2_damping),
            "torque_scale_1": float(args.torque_scale_1),
            "torque_scale_2": float(args.torque_scale_2),
            "gravity": float(args.gravity),
        },
        "learned_params": learned_params,
        "model_config": {
            "use_rk4": args.use_rk4,
            "mass_matrix_reg": args.mass_matrix_reg,
        },
        "training_config": {
            "dt": float(args.dt),
            "target_weight": float(args.target_weight),
            "lr": args.lr,
            "grad_clip": args.grad_clip,
        },
        "train_losses": train_losses,
        "val_losses": val_losses,
        "normalization_std": normalization_std.cpu(),  # Save for future evaluation
    }

    if args.buffer_path:
        ckpt["buffer_path"] = args.buffer_path
    else:
        ckpt["buffer_paths"] = args.buffer_paths

    torch.save(ckpt, args.save_model)
    print(f"Model checkpoint saved to: {args.save_model}")
    print("\nTraining Summary:")
    print(f"  Final Train Loss: {train_losses[-1]:.6f}")
    print(f"  Final Val Loss:   {val_losses[-1]:.6f}")
    print(f"  Best Train Loss:  {min(train_losses):.6f}")
    print(f"  Best Val Loss:    {min(val_losses):.6f}")

    # Print comparison of initial vs learned parameters
    if learned_params:
        print("\nParameter Learning Summary (Initial → Learned):")
        print("=" * 60)
        initial = ckpt["initial_params"]

        # Map between initial_params keys and learned_params keys
        key_mapping = {
            "mass_link1": "m1",
            "mass_link2": "m2",
            "length_link1": "l1",
            "length_link2": "l2",
            "com_link1": "lc1",
            "com_link2": "lc2",
            "inertia_link1": "I1",
            "inertia_link2": "I2",
            "joint1_damping": "b1",
            "joint2_damping": "b2",
            "torque_scale_1": "torque_scale_1",
            "torque_scale_2": "torque_scale_2",
            "gravity": "g",
        }

        for init_key, learned_key in key_mapping.items():
            init_val = initial[init_key]
            learned_val = learned_params[learned_key]
            change_pct = (
                ((learned_val - init_val) / init_val * 100)
                if init_val != 0
                else float("inf")
            )
            print(
                f"  {init_key:20s}: {init_val:10.6f} → {learned_val:10.6f} ({change_pct:+7.2f}%)"
            )

    print("=" * 60)


if __name__ == "__main__":
    main()

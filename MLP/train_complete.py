#!/usr/bin/env python3
"""
Complete Behavioral Cloning Training Script with All Features
============================================================

This script combines all training features into one comprehensive tool:
- Single buffer training
- Multi-buffer training
- Reward-weighted behavioral cloning
- Online testing
- Multiple reward weighting methods
- Comprehensive visualization

Usage Examples:
    # Train on single buffer
    python train_complete.py --buffer_path buffer.pt

    # Train on multiple buffers
    python train_complete.py --buffer_paths buffer1.pt buffer2.pt

    # Train with online testing
    python train_complete.py --buffer_path buffer.pt --test_online

    # Train with custom reward weighting
    python train_complete.py --buffer_path buffer.pt --reward_weighting threshold --reward_scale 5.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from tqdm import tqdm

# import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np
import glob
import os
from types import SimpleNamespace

# Add the parent directory to the path to import tdmpc2 modules
sys.path.append(str(Path(__file__).parent.parent))

from tdmpc2.common.layers import mlp


class RewardWeightedBehavioralCloningMLP(nn.Module):
    """
    MLP for behavioral cloning that considers rewards.
    Maps observations to actions, with reward-weighted loss.
    """

    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256], dropout=0.1):
        super().__init__()
        self.obs_dim = obs_dim
        print(f"Obs dim: {obs_dim}")
        self.action_dim = action_dim
        print(f"Action dim: {action_dim}")
        # Use the existing mlp function from tdmpc2
        self.network = mlp(
            in_dim=obs_dim,
            mlp_dims=hidden_dims,
            out_dim=action_dim,
            act=None,  # No activation on output layer
            dropout=dropout,
        )

    def forward(self, obs):
        return self.network(obs)


def load_single_buffer(buffer_path, device="cpu"):
    """Load a single buffer file."""
    print(f"Loading single buffer: {buffer_path}")
    buffer_data = torch.load(buffer_path, weights_only=False)

    # Check if it's a TensorDict
    if hasattr(buffer_data, "shape") and hasattr(buffer_data, "keys"):
        print(f"Buffer shape: {buffer_data.shape}")
        print(f"Buffer keys: {list(buffer_data.keys())}")

        # For TensorDict, the shape represents the batch dimensions
        # buffer_data.shape could be (num_episodes, episode_length) or (total_steps,)
        return buffer_data
    else:
        # Handle other formats if needed
        print(f"Buffer type: {type(buffer_data)}")
        return buffer_data


def load_multiple_buffers(buffer_paths, device="cpu"):
    """
    Load and combine multiple buffer files.

    Args:
        buffer_paths: List of paths to buffer files or directory containing buffers
        device: Device to move data to

    Returns:
        combined_buffer_data: Combined TensorDict from all buffers
    """
    print("Loading multiple buffers...")

    # Handle different input types
    if isinstance(buffer_paths, str):
        # Single path - could be directory or file
        if os.path.isdir(buffer_paths):
            # Directory - find all .pt files
            buffer_files = glob.glob(os.path.join(buffer_paths, "*.pt"))
            buffer_files = [f for f in buffer_files if "metadata" not in f]
            print(
                f"Found {len(buffer_files)} buffer files in directory: {buffer_paths}"
            )
        else:
            # Single file
            buffer_files = [buffer_paths]
    else:
        # List of paths
        buffer_files = buffer_paths

    if not buffer_files:
        raise ValueError("No buffer files found!")

    print(f"Loading {len(buffer_files)} buffer files:")
    for i, file_path in enumerate(buffer_files):
        print(f"  {i + 1}. {file_path}")

    # Load and combine all buffers
    all_episodes = []
    total_episodes = 0
    total_steps = 0

    for i, buffer_file in enumerate(buffer_files):
        print(f"\nLoading buffer {i + 1}/{len(buffer_files)}: {buffer_file}")

        try:
            buffer_data = torch.load(buffer_file, weights_only=False)

            # Check if it's a TensorDict
            if hasattr(buffer_data, "shape") and hasattr(buffer_data, "keys"):
                print(f"  Buffer shape: {buffer_data.shape}")
                print(f"  Buffer keys: {list(buffer_data.keys())}")

                # For TensorDict, shape could be:
                # - (num_episodes, episode_length) for episodic data
                # - (total_steps,) for flattened data

                if len(buffer_data.shape) == 2:
                    # Episodic format: (num_episodes, episode_length)
                    num_episodes, episode_length = buffer_data.shape
                    all_episodes.append(buffer_data)
                    total_episodes += num_episodes
                    total_steps += num_episodes * episode_length
                    print(
                        f"  Episodes: {num_episodes}, Episode length: {episode_length}"
                    )

                elif len(buffer_data.shape) == 1:
                    # Flattened format: (total_steps,)
                    total_steps_in_buffer = buffer_data.shape[0]
                    all_episodes.append(buffer_data)
                    total_steps += total_steps_in_buffer
                    # For tracking, we'll count this as individual steps rather than episodes
                    print(f"  Total steps: {total_steps_in_buffer}")

                else:
                    print(f"  Warning: Unexpected buffer shape: {buffer_data.shape}")
                    # Try to use it anyway
                    all_episodes.append(buffer_data)

            else:
                # Handle other buffer formats
                print(f"  Buffer type: {type(buffer_data)}")
                if hasattr(buffer_data, "shape"):
                    print(f"  Buffer shape: {buffer_data.shape}")
                all_episodes.append(buffer_data)

        except Exception as e:
            print(f"  Error loading {buffer_file}: {e}")
            continue

    if not all_episodes:
        raise ValueError("No valid buffers were loaded!")

    # Combine all episodes
    print(f"\nCombining {len(all_episodes)} buffers...")

    # For TensorDict, we need to concatenate along the appropriate dimension
    # If all buffers have the same structure, we can concatenate them
    try:
        if len(all_episodes) == 1:
            combined_buffer = all_episodes[0]
        else:
            # Try to concatenate along the first dimension
            # This works for both episodic and flattened formats
            from tensordict import TensorDict

            if all(isinstance(ep, TensorDict) for ep in all_episodes):
                # Use TensorDict's cat method
                combined_buffer = TensorDict.cat(all_episodes, dim=0)
            else:
                # Fallback to torch.cat if not TensorDict
                combined_buffer = torch.cat(all_episodes, dim=0)

        print(f"Combined buffer shape: {combined_buffer.shape}")
        print(f"Combined buffer keys: {list(combined_buffer.keys())}")
        print(
            f"Total episodes/steps processed: {total_episodes if total_episodes > 0 else total_steps}"
        )

    except Exception as e:
        print(f"Error combining buffers: {e}")
        # If concatenation fails, return the first buffer as fallback
        print("Using first buffer as fallback...")
        combined_buffer = all_episodes[0]

    return combined_buffer


def process_buffer_data(
    buffer_data, device="cpu", reward_weighting="exponential", reward_scale=1.0
):
    """
    Process TDMPC2 buffer data for reward-weighted behavioral cloning training.

    Args:
        buffer_data: TensorDict containing buffer data
        device: Device to move data to
        reward_weighting: Method for reward weighting ('exponential', 'linear', 'threshold')
        reward_scale: Scale factor for reward weighting

    Returns:
        obs_tensor: Processed observations
        action_tensor: Processed actions
        reward_weights: Weights for each sample based on rewards
    """
    print("Processing buffer data with reward weighting...")
    print(f"Buffer type: {type(buffer_data)}")
    print(f"Buffer shape: {buffer_data.shape}")
    print(f"Buffer keys: {list(buffer_data.keys())}")

    # Extract observations, actions, and rewards from TensorDict
    obs = buffer_data["obs"]
    actions = buffer_data["action"]
    rewards = buffer_data["reward"]

    print(f"Raw obs shape: {obs.shape}")
    print(f"Raw action shape: {actions.shape}")
    print(f"Raw reward shape: {rewards.shape}")

    # Print sample data statistics
    print(
        f"Sample obs stats: mean={obs.float().mean():.3f}, std={obs.float().std():.3f}"
    )
    print(
        f"Sample action stats: mean={actions.float().mean():.3f}, std={actions.float().std():.3f}"
    )
    print(
        f"Sample reward stats: mean={rewards.float().mean():.3f}, std={rewards.float().std():.3f}"
    )

    # Handle different buffer structures
    # TensorDict shapes can be:
    # - (num_episodes, episode_length, feature_dim) for episodic data
    # - (total_steps, feature_dim) for flattened data

    if len(obs.shape) == 3:
        # Shape: (num_episodes, episode_length, obs_dim)
        # Flatten to (total_steps, obs_dim)
        obs = obs.view(-1, obs.shape[-1])
        actions = actions.view(-1, actions.shape[-1])
        rewards = rewards.view(-1, rewards.shape[-1] if len(rewards.shape) > 2 else 1)

    elif len(obs.shape) == 2:
        # Shape: (total_steps, obs_dim) - already flattened
        # Ensure rewards have correct shape
        if len(rewards.shape) == 1:
            rewards = rewards.unsqueeze(-1)  # Add feature dimension
        elif len(rewards.shape) > 2:
            rewards = rewards.view(-1, rewards.shape[-1])

    else:
        raise ValueError(f"Unexpected observation shape: {obs.shape}")

    # Ensure actions have correct shape
    if len(actions.shape) == 1:
        actions = actions.unsqueeze(-1)  # Add feature dimension
    elif len(actions.shape) > 2:
        actions = actions.view(-1, actions.shape[-1])

    print(f"Processed obs shape: {obs.shape}")
    print(f"Processed action shape: {actions.shape}")
    print(f"Processed reward shape: {rewards.shape}")

    # Check for NaN/inf values in data
    obs_has_nan = torch.isnan(obs).any() or torch.isinf(obs).any()
    actions_has_nan = torch.isnan(actions).any() or torch.isinf(actions).any()
    rewards_has_nan = torch.isnan(rewards).any() or torch.isinf(rewards).any()

    if obs_has_nan:
        print("WARNING: Found NaN/inf values in observations!")
        # Replace NaN/inf with 0
        obs = torch.where(torch.isnan(obs) | torch.isinf(obs), torch.tensor(0.0), obs)

    if actions_has_nan:
        print("WARNING: Found NaN/inf values in actions!")
        # Replace NaN/inf with 0
        actions = torch.where(
            torch.isnan(actions) | torch.isinf(actions), torch.tensor(0.0), actions
        )

    if rewards_has_nan:
        print("WARNING: Found NaN/inf values in rewards!")
        # Replace NaN/inf with 0
        rewards = torch.where(
            torch.isnan(rewards) | torch.isinf(rewards), torch.tensor(0.0), rewards
        )

    # Convert to float32 and move to device
    obs_tensor = obs.float().to(device)
    action_tensor = actions.float().to(device)
    reward_tensor = rewards.float().to(device)

    # Compute reward weights
    reward_weights = compute_reward_weights(
        reward_tensor, reward_weighting, reward_scale
    )

    # Check if weights have NaN values
    if torch.isnan(reward_weights).any() or torch.isinf(reward_weights).any():
        print("WARNING: Found NaN/inf values in reward weights! Using uniform weights.")
        reward_weights = torch.ones_like(reward_weights)

    # Basic data validation
    print(f"Observation range: [{obs_tensor.min():.3f}, {obs_tensor.max():.3f}]")
    print(f"Action range: [{action_tensor.min():.3f}, {action_tensor.max():.3f}]")
    print(f"Reward range: [{reward_tensor.min():.3f}, {reward_tensor.max():.3f}]")
    print(
        f"Reward weight range: [{reward_weights.min():.3f}, {reward_weights.max():.3f}]"
    )
    print(f"Number of training samples: {len(obs_tensor)}")

    # Final validation - check for any remaining NaN/inf
    if (
        torch.isnan(obs_tensor).any()
        or torch.isnan(action_tensor).any()
        or torch.isnan(reward_weights).any()
    ):
        raise ValueError("Data still contains NaN values after cleaning!")

    return obs_tensor, action_tensor, reward_weights


def compute_reward_weights(rewards, method="exponential", scale=1.0):
    """
    Compute weights for each sample based on rewards.

    Args:
        rewards: Tensor of rewards
        method: Weighting method ('exponential', 'linear', 'threshold')
        scale: Scale factor for weighting

    Returns:
        weights: Tensor of weights for each sample
    """
    rewards = rewards.squeeze()  # Remove extra dimensions

    if method == "exponential":
        # Exponential weighting: higher rewards get exponentially higher weights
        # Normalize rewards to [0, 1] first, then apply exponential
        min_reward = rewards.min()
        max_reward = rewards.max()
        if max_reward > min_reward:
            normalized_rewards = (rewards - min_reward) / (max_reward - min_reward)
        else:
            normalized_rewards = torch.ones_like(rewards)

        # Apply exponential scaling
        weights = torch.exp(scale * normalized_rewards)

    elif method == "linear":
        # Linear weighting: higher rewards get proportionally higher weights
        min_reward = rewards.min()
        max_reward = rewards.max()
        if max_reward > min_reward:
            weights = (rewards - min_reward) / (max_reward - min_reward) * scale + 1.0
        else:
            weights = torch.ones_like(rewards)

    elif method == "threshold":
        # Threshold weighting: only samples above threshold get weight
        threshold = rewards.quantile(0.7)  # Top 30% of rewards
        weights = torch.where(
            rewards >= threshold, torch.tensor(scale), torch.tensor(1.0)
        )

    else:
        raise ValueError(f"Unknown reward weighting method: {method}")

    # Normalize weights to have mean 1.0 to avoid changing the overall loss scale
    weights = weights / weights.mean()

    return weights


def create_data_loaders(
    obs_tensor, action_tensor, reward_weights, batch_size=256, train_split=0.8
):
    """
    Create training and validation data loaders with reward weights.

    Args:
        obs_tensor: Observation tensor
        action_tensor: Action tensor
        reward_weights: Reward-based weights for each sample
        batch_size: Batch size for training
        train_split: Fraction of data to use for training

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    # Create dataset with weights
    dataset = TensorDataset(obs_tensor, action_tensor, reward_weights)

    # Split into train/val
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-3, device="cpu"):
    """
    Train the reward-weighted behavioral cloning model.

    Args:
        model: The MLP model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on

    Returns:
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="none")  # No reduction for weighted loss

    train_losses = []
    val_losses = []

    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0

        for obs_batch, action_batch, weight_batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            weight_batch = weight_batch.to(device)

            optimizer.zero_grad()
            pred_actions = model(obs_batch)

            # Compute element-wise loss
            element_loss = criterion(pred_actions, action_batch)

            # Apply reward weights (mean over action dimensions, then weight)
            if len(element_loss.shape) > 1:
                element_loss = element_loss.mean(dim=1)  # Mean over action dimensions

            # Weighted loss
            weighted_loss = (element_loss * weight_batch).mean()

            weighted_loss.backward()
            optimizer.step()

            train_loss += weighted_loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for obs_batch, action_batch, weight_batch in val_loader:
                obs_batch = obs_batch.to(device)
                action_batch = action_batch.to(device)
                weight_batch = weight_batch.to(device)

                pred_actions = model(obs_batch)
                element_loss = criterion(pred_actions, action_batch)

                if len(element_loss.shape) > 1:
                    element_loss = element_loss.mean(dim=1)

                weighted_loss = (element_loss * weight_batch).mean()

                val_loss += weighted_loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}"
            )

    return train_losses, val_losses


# def plot_training_curves(train_losses, val_losses, save_path=None):
#     """Plot training and validation loss curves."""
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label="Training Loss", alpha=0.8)
#     plt.plot(val_losses, label="Validation Loss", alpha=0.8)
#     plt.xlabel("Epoch")
#     plt.ylabel("Weighted MSE Loss")
#     plt.title("Reward-Weighted Behavioral Cloning Training Curves")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"Training curves saved to: {save_path}")

#     plt.show()


def test_model_online(
    model, env_name="inverted-double-pendulum", num_episodes=10, device="cpu"
):
    """
    Test the trained model online in the MuJoCo environment.

    Args:
        model: Trained behavioral cloning model
        env_name: Name of the environment to test on
        num_episodes: Number of episodes to run
        device: Device to run inference on

    Returns:
        episode_rewards: List of total rewards for each episode
        episode_lengths: List of episode lengths
    """
    print(f"Testing model online on {env_name} for {num_episodes} episodes...")

    # Create environment configuration
    cfg = SimpleNamespace()
    cfg.task = env_name
    cfg.obs = "state"
    cfg.multitask = False

    # Import and create environment with correct import path
    try:
        # Import from the correct path within tdmpc2
        from tdmpc2.envs.mujoco import make_env

        env = make_env(cfg)
        print(f"Environment created: {env}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Check action space shape
        expected_action_shape = env.action_space.shape
        print(f"Expected action shape: {expected_action_shape}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Attempting fallback environment creation...")

        # Fallback: try creating a simple gymnasium environment
        try:
            import gymnasium as gym

            env_map = {
                "inverted-double-pendulum": "InvertedDoublePendulum-v4",
                "inverted-pendulum": "InvertedPendulum-v4",
                "half-cheetah": "HalfCheetah-v4",
                "walker": "Walker2d-v4",
                "hopper": "Hopper-v4",
            }

            gym_env_name = env_map.get(env_name, "InvertedDoublePendulum-v4")
            env = gym.make(gym_env_name)
            print(f"Created fallback gymnasium environment: {gym_env_name}")
            print(f"Observation space: {env.observation_space}")
            print(f"Action space: {env.action_space}")

            # Check action space shape
            expected_action_shape = env.action_space.shape
            print(f"Expected action shape: {expected_action_shape}")
        except Exception as e2:
            print(f"Fallback environment creation also failed: {e2}")
            return [], []

    model.eval()
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        try:
            # Reset environment
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result  # Gymnasium v0.26+ returns (obs, info)
            else:
                obs = reset_result  # Older versions return just obs

            total_reward = 0
            episode_length = 0
            done = False

            print(f"Episode {episode + 1}/{num_episodes}: ", end="", flush=True)

            while not done:
                # Convert observation to tensor and get action from model
                if isinstance(obs, dict):
                    # Handle dict observations (take 'state' key or first key)
                    obs_key = "state" if "state" in obs else list(obs.keys())[0]
                    obs_array = obs[obs_key]
                else:
                    obs_array = obs

                obs_tensor = (
                    torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0).to(device)
                )

                with torch.no_grad():
                    action_tensor = model(obs_tensor)
                    action = (
                        action_tensor.cpu().numpy().squeeze(0)
                    )  # Only remove batch dimension

                    # Debug: print action shape for first episode
                    if episode == 0 and episode_length == 0:
                        print(
                            f"Model output shape: {action_tensor.shape}, Action shape: {action.shape}"
                        )

                    # Ensure action has correct shape for environment
                    if hasattr(env, "action_space"):
                        expected_shape = env.action_space.shape
                        if action.shape != expected_shape:
                            if len(expected_shape) == 1 and action.ndim == 0:
                                # Convert scalar to 1D array
                                action = np.array([action])
                            elif (
                                len(expected_shape) == 1
                                and action.ndim == 1
                                and action.shape[0] != expected_shape[0]
                            ):
                                # Pad or truncate to match expected size
                                if action.shape[0] < expected_shape[0]:
                                    action = np.pad(
                                        action, (0, expected_shape[0] - action.shape[0])
                                    )
                                else:
                                    action = action[: expected_shape[0]]

                # Take step in environment
                step_result = env.step(action)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated

                total_reward += reward
                episode_length += 1

                # Safety check to prevent infinite episodes
                if episode_length > 2000:
                    print("Episode too long, terminating...")
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            print(f"Reward: {total_reward:.2f}, Length: {episode_length}")

        except Exception as e:
            print(f"Error in episode {episode + 1}: {e}")
            episode_rewards.append(0.0)
            episode_lengths.append(0)
            continue

    try:
        env.close()
    except Exception:
        pass

    # Print statistics
    if episode_rewards:
        print("\nOnline Testing Results:")
        print(
            f"  Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
        )
        print(
            f"  Average episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}"
        )
        print(f"  Best episode reward: {np.max(episode_rewards):.2f}")
        print(f"  Worst episode reward: {np.min(episode_rewards):.2f}")
    else:
        print("No episodes completed successfully.")

    return episode_rewards, episode_lengths


# def plot_online_results(episode_rewards, episode_lengths, save_path=None):
#     """Plot online testing results."""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

#     # Plot episode rewards
#     ax1.plot(episode_rewards, "o-", alpha=0.7)
#     ax1.axhline(
#         y=np.mean(episode_rewards),
#         color="r",
#         linestyle="--",
#         label=f"Mean: {np.mean(episode_rewards):.2f}",
#     )
#     ax1.set_xlabel("Episode")
#     ax1.set_ylabel("Total Reward")
#     ax1.set_title("Online Testing: Episode Rewards")
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)

#     # Plot episode lengths
#     ax2.plot(episode_lengths, "o-", alpha=0.7, color="green")
#     ax2.axhline(
#         y=np.mean(episode_lengths),
#         color="r",
#         linestyle="--",
#         label=f"Mean: {np.mean(episode_lengths):.2f}",
#     )
#     ax2.set_xlabel("Episode")
#     ax2.set_ylabel("Episode Length")
#     ax2.set_title("Online Testing: Episode Lengths")
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"Online testing results saved to: {save_path}")

#     plt.show()


def load_and_test_model(
    model_path, env_name="inverted-double-pendulum", num_episodes=10, device="cpu"
):
    """
    Load a trained model and test it online.

    Args:
        model_path: Path to the saved model
        env_name: Name of the environment to test on
        num_episodes: Number of episodes to run
        device: Device to run inference on
    """
    print(f"Loading model from: {model_path}")

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create model with same architecture
    model = RewardWeightedBehavioralCloningMLP(
        obs_dim=checkpoint["obs_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"],
    )

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print("Model loaded successfully!")
    print(f"  Observation dimension: {checkpoint['obs_dim']}")
    print(f"  Action dimension: {checkpoint['action_dim']}")
    print(f"  Hidden dimensions: {checkpoint['hidden_dims']}")
    print(f"  Reward weighting: {checkpoint.get('reward_weighting', 'unknown')}")
    print(f"  Reward scale: {checkpoint.get('reward_scale', 'unknown')}")

    # Test model online
    episode_rewards, episode_lengths = test_model_online(
        model, env_name, num_episodes, device
    )

    # Plot results
    # plot_online_results(episode_rewards, episode_lengths, "online_testing_results.png")

    return episode_rewards, episode_lengths


def demonstrate_reward_weighting():
    """Demonstrate different reward weighting methods."""
    print("=" * 60)
    print("REWARD WEIGHTING DEMONSTRATION")
    print("=" * 60)

    # Create example rewards
    rewards = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]).unsqueeze(1)
    print(f"Example rewards: {rewards.squeeze().numpy()}")

    # Test different weighting methods
    methods = ["exponential", "linear", "threshold"]
    scales = [1.0, 2.0, 3.0]

    for method in methods:
        print(f"\n{method.upper()} weighting:")
        for scale in scales:
            weights = compute_reward_weights(rewards, method, scale)
            print(f"  Scale {scale}: {weights.squeeze().numpy()}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete Behavioral Cloning Training with All Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on single buffer
    python train_complete.py --buffer_path buffer.pt
    
    # Train on multiple buffers
    python train_complete.py --buffer_paths buffer1.pt buffer2.pt
    
    # Train with online testing
    python train_complete.py --buffer_path buffer.pt --test_online
    
    # Train with custom reward weighting
    python train_complete.py --buffer_path buffer.pt --reward_weighting threshold --reward_scale 5.0
    
    # Test a saved model
    python train_complete.py --test_only --model_path model.pt --test_episodes 20
        """,
    )

    # Buffer input options (mutually exclusive)
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

    # Training parameters
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
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )

    # Model saving
    parser.add_argument(
        "--save_model",
        type=str,
        default="bc_mlp_model.pt",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--save_plots",
        type=str,
        default="training_curves.png",
        help="Path to save training curves plot",
    )

    # Reward weighting
    parser.add_argument(
        "--reward_weighting",
        type=str,
        default="exponential",
        choices=["exponential", "linear", "threshold"],
        help="Method for reward weighting",
    )
    parser.add_argument(
        "--reward_scale",
        type=float,
        default=2.0,
        help="Scale factor for reward weighting",
    )

    # Online testing
    parser.add_argument(
        "--test_online",
        action="store_true",
        help="Test the trained model online after training",
    )
    parser.add_argument(
        "--test_episodes",
        type=int,
        default=10,
        help="Number of episodes for online testing",
    )
    parser.add_argument(
        "--test_env",
        type=str,
        default="inverted-double-pendulum",
        choices=[
            "inverted-double-pendulum",
            "mujoco-walker",
            "mujoco-halfcheetah",
            "bipedal-walker",
            "lunarlander-continuous",
        ],
        help="Environment name for online testing",
    )

    # Test only mode
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only test a saved model (skip training)",
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to saved model for testing only"
    )

    # Utility options
    parser.add_argument(
        "--demo_reward_weighting",
        action="store_true",
        help="Demonstrate reward weighting methods and exit",
    )

    args = parser.parse_args()

    # Handle demo mode
    if args.demo_reward_weighting:
        demonstrate_reward_weighting()
        return

    # Handle test-only mode
    if args.test_only:
        if not args.model_path:
            parser.error("--model_path is required when using --test_only")

        # Set device
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        print(f"Testing model: {args.model_path}")
        print(f"Environment: {args.test_env}")
        print(f"Episodes: {args.test_episodes}")
        print(f"Device: {device}")

        # Load and test model
        episode_rewards, episode_lengths = load_and_test_model(
            args.model_path, args.test_env, args.test_episodes, device
        )

        print("\nFinal Results:")
        print(f"  Average reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
        print(
            f"  Average episode length: {sum(episode_lengths) / len(episode_lengths):.2f}"
        )
        print(f"  Best episode: {max(episode_rewards):.2f}")
        print(f"  Worst episode: {min(episode_rewards):.2f}")
        return

    # Training mode - require buffer input
    if not args.buffer_path and not args.buffer_paths:
        parser.error("Either --buffer_path or --buffer_paths is required for training")

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Reward weighting method: {args.reward_weighting}")
    print(f"Reward scale: {args.reward_scale}")

    # Load buffer data
    if args.buffer_path:
        print(f"Loading single buffer: {args.buffer_path}")
        buffer_data = load_single_buffer(args.buffer_path, device)
    else:
        print(f"Loading multiple buffers: {args.buffer_paths}")
        buffer_data = load_multiple_buffers(args.buffer_paths, device)

    # Process buffer data with reward weighting
    obs_tensor, action_tensor, reward_weights = process_buffer_data(
        buffer_data, device, args.reward_weighting, args.reward_scale
    )

    # Get dimensions
    obs_dim = obs_tensor.shape[1]
    print
    action_dim = action_tensor.shape[1]

    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")

    # Create model
    model = RewardWeightedBehavioralCloningMLP(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        obs_tensor, action_tensor, reward_weights, batch_size=args.batch_size
    )

    # Train model
    print("Starting reward-weighted training...")
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    # Save model
    model_info = {
        "model_state_dict": model.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "reward_weighting": args.reward_weighting,
        "reward_scale": args.reward_scale,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    if args.buffer_path:
        model_info["buffer_path"] = args.buffer_path
    else:
        model_info["buffer_paths"] = args.buffer_paths

    torch.save(model_info, args.save_model)
    print(f"Model saved to: {args.save_model}")

    # Plot training curves
    print(f"Train losses: {train_losses[-1]:.4f}")
    print(f"Val losses: {val_losses[-1]:.4f}")
    # plot_training_curves(train_losses, val_losses, args.save_plots)

    # Test model online if requested
    if args.test_online:
        print("\n" + "=" * 60)
        print("ONLINE TESTING")
        print("=" * 60)
        episode_rewards, episode_lengths = test_model_online(
            model, args.test_env, args.test_episodes, device
        )
        print(f"Episode rewards: {episode_rewards}")
        print(f"Episode lengths: {episode_lengths}")
        # plot_online_results(
        #     episode_rewards, episode_lengths, "online_testing_results.png"
        # )

    print("Training completed!")


if __name__ == "__main__":
    main()

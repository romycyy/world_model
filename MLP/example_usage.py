#!/usr/bin/env python3
"""
Example usage of the reward-weighted behavioral cloning training script.
This script demonstrates how to train a behavioral cloning model that prioritizes
high-reward actions from TDMPC2 buffer data.
"""

import torch
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import tdmpc2 modules
sys.path.append(str(Path(__file__).parent.parent))

from train import (
    RewardWeightedBehavioralCloningMLP, 
    process_buffer_data, 
    create_data_loaders, 
    train_model, 
    plot_training_curves,
    compute_reward_weights
)


def create_example_buffer_data():
    """Create example buffer data with varying rewards for demonstration."""
    from tensordict.tensordict import TensorDict
    
    # Create synthetic data similar to TDMPC2 buffer format
    num_episodes = 5
    episode_length = 50
    obs_dim = 8
    action_dim = 2
    
    # Create synthetic episodes with varying reward patterns
    episodes = []
    for ep_idx in range(num_episodes):
        # Create synthetic observations and actions
        obs = torch.randn(episode_length, obs_dim)
        action = torch.randn(episode_length, action_dim)
        
        # Create rewards with different patterns to demonstrate weighting
        if ep_idx == 0:
            # High rewards throughout
            reward = torch.ones(episode_length, 1) * 2.0 + torch.randn(episode_length, 1) * 0.1
        elif ep_idx == 1:
            # Low rewards throughout
            reward = torch.ones(episode_length, 1) * -1.0 + torch.randn(episode_length, 1) * 0.1
        elif ep_idx == 2:
            # Increasing rewards (good episode)
            reward = torch.linspace(-1.0, 2.0, episode_length).unsqueeze(1) + torch.randn(episode_length, 1) * 0.1
        elif ep_idx == 3:
            # Decreasing rewards (bad episode)
            reward = torch.linspace(2.0, -1.0, episode_length).unsqueeze(1) + torch.randn(episode_length, 1) * 0.1
        else:
            # Random rewards
            reward = torch.randn(episode_length, 1)
        
        terminated = torch.zeros(episode_length, 1)
        terminated[-1] = 1.0  # Mark last step as terminated
        episode = torch.full((episode_length,), ep_idx, dtype=torch.long)
        
        # Create TensorDict for this episode
        td = TensorDict(
            {
                "episode": episode,
                "obs": obs,
                "action": action,
                "reward": reward,
                "terminated": terminated,
            },
            batch_size=(episode_length,),
        )
        
        episodes.append(td)
    
    # Stack all episodes into a single TensorDict
    buffer_data = torch.stack(episodes, dim=0)
    return buffer_data


def demonstrate_reward_weighting():
    """Demonstrate different reward weighting methods."""
    print("=" * 60)
    print("REWARD WEIGHTING DEMONSTRATION")
    print("=" * 60)
    
    # Create example rewards
    rewards = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]).unsqueeze(1)
    print(f"Example rewards: {rewards.squeeze().numpy()}")
    
    # Test different weighting methods
    methods = ['exponential', 'linear', 'threshold']
    scales = [1.0, 2.0, 3.0]
    
    for method in methods:
        print(f"\n{method.upper()} weighting:")
        for scale in scales:
            weights = compute_reward_weights(rewards, method, scale)
            print(f"  Scale {scale}: {weights.squeeze().numpy()}")


def main():
    """Example usage of the reward-weighted behavioral cloning training."""
    print("=" * 60)
    print("REWARD-WEIGHTED BEHAVIORAL CLONING EXAMPLE")
    print("=" * 60)
    
    # Demonstrate reward weighting methods
    demonstrate_reward_weighting()
    
    # Create example buffer data
    print("\n" + "=" * 60)
    print("TRAINING EXAMPLE")
    print("=" * 60)
    print("Creating example buffer data with varying rewards...")
    buffer_data = create_example_buffer_data()
    
    # Process the buffer data with reward weighting
    print("\nProcessing buffer data with exponential reward weighting...")
    obs_tensor, action_tensor, reward_weights = process_buffer_data(
        buffer_data, device='cpu', reward_weighting='exponential', reward_scale=2.0
    )
    
    # Get dimensions
    obs_dim = obs_tensor.shape[1]
    action_dim = action_tensor.shape[1]
    
    print(f"\nData dimensions:")
    print(f"  Observation dimension: {obs_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Total samples: {len(obs_tensor)}")
    
    # Show reward weight statistics
    print(f"\nReward weight statistics:")
    print(f"  Mean weight: {reward_weights.mean():.3f}")
    print(f"  Std weight: {reward_weights.std():.3f}")
    print(f"  Min weight: {reward_weights.min():.3f}")
    print(f"  Max weight: {reward_weights.max():.3f}")
    
    # Create model
    print(f"\nCreating reward-weighted MLP model...")
    model = RewardWeightedBehavioralCloningMLP(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=[128, 128],  # Smaller for demo
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    print(f"\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(
        obs_tensor, action_tensor, reward_weights,
        batch_size=32,  # Smaller batch for demo
        train_split=0.8
    )
    
    # Train model (shorter training for demo)
    print(f"\nStarting reward-weighted training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        num_epochs=20,  # Fewer epochs for demo
        lr=1e-3,
        device='cpu'
    )
    
    # Plot training curves
    print(f"\nPlotting training curves...")
    plot_training_curves(train_losses, val_losses, 'example_reward_weighted_training_curves.png')
    
    # Test the trained model
    print(f"\nTesting trained model...")
    model.eval()
    with torch.no_grad():
        # Get a batch from validation set
        obs_batch, action_batch, weight_batch = next(iter(val_loader))
        pred_actions = model(obs_batch)
        
        # Calculate test loss
        mse_loss = torch.nn.functional.mse_loss(pred_actions, action_batch)
        print(f"Test MSE Loss: {mse_loss.item():.6f}")
        
        # Show some predictions vs ground truth with their weights
        print(f"\nSample predictions vs ground truth (with reward weights):")
        for i in range(min(3, len(obs_batch))):
            print(f"  Sample {i+1} (weight: {weight_batch[i]:.3f}):")
            print(f"    Predicted: {pred_actions[i].numpy()}")
            print(f"    Ground truth: {action_batch[i].numpy()}")
    
    # Compare with uniform weighting
    print(f"\n" + "=" * 60)
    print("COMPARISON: UNIFORM vs REWARD-WEIGHTED")
    print("=" * 60)
    
    # Create uniform weights for comparison
    uniform_weights = torch.ones_like(reward_weights)
    uniform_train_loader, uniform_val_loader = create_data_loaders(
        obs_tensor, action_tensor, uniform_weights,
        batch_size=32, train_split=0.8
    )
    
    # Train uniform model
    print("Training model with uniform weighting...")
    uniform_model = RewardWeightedBehavioralCloningMLP(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dims=[128, 128], dropout=0.1
    )
    
    uniform_train_losses, uniform_val_losses = train_model(
        uniform_model, uniform_train_loader, uniform_val_loader,
        num_epochs=20, lr=1e-3, device='cpu'
    )
    
    # Compare final losses
    print(f"\nFinal validation losses:")
    print(f"  Reward-weighted model: {val_losses[-1]:.6f}")
    print(f"  Uniform-weighted model: {uniform_val_losses[-1]:.6f}")
    
    print(f"\nExample completed successfully!")
    print(f"Training curves saved to: example_reward_weighted_training_curves.png")


if __name__ == "__main__":
    main()

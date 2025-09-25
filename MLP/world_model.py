import torch
import torch.nn as nn
from typing import Iterable, List, Optional, Type


class WorldModelMLP(nn.Module):
    """
    Vanilla MLP world model that predicts the next observation given the current
    observation and action.

    The model concatenates observation and action, runs them through an MLP,
    and outputs either the next observation directly or a delta that is added
    to the current observation to form the next observation.

    Args:
        obs_dim: Dimension of the observation/state vector
        action_dim: Dimension of the action vector
        hidden_dims: Sizes of hidden layers
        dropout: Dropout probability applied after each hidden layer (0.0 disables)
        predict_delta: If True, the network predicts state deltas; the forward
            returns obs + delta. If False, it predicts the next observation directly.
        activation: Activation module class to use between layers (default: nn.ReLU)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Optional[Iterable[int]] = None,
        dropout: float = 0.0,
        predict_delta: bool = False,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.predict_delta = bool(predict_delta)

        layer_dims: List[int] = [self.obs_dim + self.action_dim, *list(hidden_dims)]

        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        # Output head produces an observation-sized vector
        layers.append(nn.Linear(layer_dims[-1], self.obs_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            observation: Tensor of shape (batch, obs_dim) or (..., obs_dim)
            action: Tensor of shape (batch, action_dim) or (..., action_dim)

        Returns:
            next_observation: Predicted next observation tensor with the same
            leading shape as inputs and last dimension obs_dim.
        """
        if observation.ndim < 2 or action.ndim < 2:
            raise ValueError(
                "Both observation and action must be at least 2D tensors: (..., dim)."
            )

        if observation.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                f"Leading dimensions must match. Got obs {observation.shape} and action {action.shape}."
            )

        if observation.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected observation last dim {self.obs_dim}, got {observation.shape[-1]}."
            )
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action last dim {self.action_dim}, got {action.shape[-1]}."
            )

        # Flatten leading dims into batch for MLP processing
        leading_shape = observation.shape[:-1]
        obs_flat = observation.reshape(-1, self.obs_dim)
        act_flat = action.reshape(-1, self.action_dim)

        x = torch.cat([obs_flat, act_flat], dim=-1)
        pred = self.network(x)

        if self.predict_delta:
            pred = obs_flat + pred

        # Restore original leading shape
        next_obs = pred.reshape(*leading_shape, self.obs_dim)
        return next_obs

    def compute_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        target_next_observation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience method to compute MSE loss between predicted and target next observations.
        """
        prediction = self.forward(observation, action)
        if prediction.shape != target_next_observation.shape:
            raise ValueError(
                f"Shape mismatch: prediction {prediction.shape} vs target {target_next_observation.shape}."
            )
        return nn.functional.mse_loss(prediction, target_next_observation)

from typing import Iterable, Optional, Type

import torch
import torch.nn as nn


class WorldModelPhysicsHybrid(nn.Module):
    """
    Hybrid symbolic + learned world model for InvertedDoublePendulum-v4.

    Inputs/outputs are identical to the vanilla MLP world model:
    - forward(observation, action) -> next_observation
      where observation has shape (..., 11) and action has shape (..., 1).

    Observation layout (per Gymnasium docs):
      [x, sin(theta1), sin(theta2), cos(theta1), cos(theta2),
       x_dot, theta1_dot, theta2_dot, cfrc_1, cfrc_2, cfrc_3]

    Model structure:
      1) Reconstruct angles (theta1, theta2) from sin/cos using atan2.
      2) Compute accelerations [x_ddot, theta1_ddot, theta2_ddot] using a
         physics-inspired basis (sin/cos terms, couplings, damping, action),
         parameterized by a small linear map with learnable coefficients. This
         encodes physical priors (periodicity, gravity-like terms, damping).
      3) Optionally add a learned residual on accelerations via a shallow MLP,
         gated by a scalar computed from features.
      4) Semi-implicit Euler integrate to obtain next generalized coordinates
         and velocities, then convert to the 11-D observation format.
      5) Predict the 3 constraint forces with a lightweight linear head.

    Notes:
      - dt is configurable; defaults to 0.02s.
      - Parameters are initialized with conservative magnitudes to ensure
        stability before learning.
    """

    def __init__(
        self,
        obs_dim: int = 11,
        action_dim: int = 1,
        dt: float = 0.02,
        use_residual: bool = True,
        residual_hidden_dims: Optional[Iterable[int]] = (8,),
        activation: Type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        if obs_dim != 11 or action_dim != 1:
            raise ValueError(
                "WorldModelPhysicsHybrid is configured for InvertedDoublePendulum-v4 with obs_dim=11 and action_dim=1."
            )

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.dt = float(dt)
        self.use_residual = bool(use_residual)

        # Learnable physical scalings and constants
        # Action gain (force scaling), gravity, and simple viscous damping
        self.action_gain = nn.Parameter(torch.tensor(10.0))  # N per unit action
        self.gravity = nn.Parameter(torch.tensor(9.81))  # m/s^2
        self.cart_damping = nn.Parameter(torch.tensor(0.05))
        self.theta1_damping = nn.Parameter(torch.tensor(0.02))
        self.theta2_damping = nn.Parameter(torch.tensor(0.02))

        # Physics-inspired basis mapping to accelerations
        # Feature vector phi(s, a) encodes periodic and coupling structure
        # Reduced basis (11 dims): [1, a, x_dot, th1_dot, th2_dot, sin th1, sin th2, cos th1, cos th2, sin(th1-th2), cos(th1-th2)]
        self._phi_dim = 11
        self._phi_scale = nn.Parameter(torch.ones(self._phi_dim))  # per-feature scaling
        self._accel_head = nn.Linear(self._phi_dim, 3, bias=True)
        self._init_accel_head()

        # Optional residual MLP that predicts accel corrections d[xddot, th1ddot, th2ddot]
        if self.use_residual:
            in_dim = self.obs_dim + self.action_dim
            layers = []
            prev = in_dim
            if residual_hidden_dims and len(tuple(residual_hidden_dims)) > 0:
                for h in residual_hidden_dims:
                    layers.append(nn.Linear(prev, h))
                    layers.append(activation())
                    prev = h
            layers.append(nn.Linear(prev, 3))
            self._residual = nn.Sequential(*layers)
            # Scalar gate for the residual based on features
            self._residual_gate = nn.Linear(self._phi_dim, 1)
        else:
            self._residual = None
            self._residual_gate = None

        # Head for constraint force prediction (3 values)
        # Lightweight linear map over [phi, accelerations]
        self._cfrc_head = nn.Linear(self._phi_dim + 3, 3)

    def _init_accel_head(self) -> None:
        # Initialize to encode weak gravity and small couplings, stable to start
        nn.init.zeros_(self._accel_head.weight)
        nn.init.zeros_(self._accel_head.bias)
        # Indices in phi for convenience for reduced basis
        SIN_TH1 = 5
        SIN_TH2 = 6
        # Encourage gravity-like behavior initially on angular accelerations
        with torch.no_grad():
            # theta1_ddot ~ -g * sin(theta1)
            self._accel_head.weight[1, SIN_TH1] = -self.gravity.item()
            # theta2_ddot ~ -g * sin(theta2)
            self._accel_head.weight[2, SIN_TH2] = -0.5 * self.gravity.item()
            # Leave cart x_ddot near 0 initially (will learn from data)

    @staticmethod
    def _split_observation(obs_flat: torch.Tensor):
        x = obs_flat[:, 0:1]
        sin_th1 = obs_flat[:, 1:2]
        sin_th2 = obs_flat[:, 2:3]
        cos_th1 = obs_flat[:, 3:4]
        cos_th2 = obs_flat[:, 4:5]
        x_dot = obs_flat[:, 5:6]
        th1_dot = obs_flat[:, 6:7]
        th2_dot = obs_flat[:, 7:8]
        cfrc = obs_flat[:, 8:11]
        return x, sin_th1, sin_th2, cos_th1, cos_th2, x_dot, th1_dot, th2_dot, cfrc

    @staticmethod
    def _angles_from_sin_cos(s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.atan2(s, c)

    def _build_phi(
        self,
        x: torch.Tensor,
        sin_th1: torch.Tensor,
        sin_th2: torch.Tensor,
        cos_th1: torch.Tensor,
        cos_th2: torch.Tensor,
        x_dot: torch.Tensor,
        th1_dot: torch.Tensor,
        th2_dot: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        th1 = self._angles_from_sin_cos(sin_th1, cos_th1)
        th2 = self._angles_from_sin_cos(sin_th2, cos_th2)  # relative angle
        th12 = th1 - th2

        a = action * self.action_gain

        # Reduced physics-informed features (11 dims)
        phi_list = [
            torch.ones_like(x),  # 0: bias
            a,  # 1: action force (scaled)
            x_dot,  # 2: cart velocity (damping)
            th1_dot,  # 3: theta1 velocity
            th2_dot,  # 4: theta2 velocity
            sin_th1,  # 5: sin(theta1)
            sin_th2,  # 6: sin(theta2)
            cos_th1,  # 7: cos(theta1)
            cos_th2,  # 8: cos(theta2)
            torch.sin(th12),  # 9: sin(theta1 - theta2)
            torch.cos(th12),  # 10: cos(theta1 - theta2)
        ]
        phi = torch.cat(phi_list, dim=-1)
        # Per-feature learnable scaling
        phi = phi * self._phi_scale
        return phi

    def _predict_accelerations(
        self,
        obs_flat: torch.Tensor,
        action_flat: torch.Tensor,
    ) -> torch.Tensor:
        x, s1, s2, c1, c2, x_dot, th1_dot, th2_dot, _ = self._split_observation(
            obs_flat
        )
        phi = self._build_phi(x, s1, s2, c1, c2, x_dot, th1_dot, th2_dot, action_flat)
        acc_base = self._accel_head(phi)

        # Add simple viscous damping explicitly to accelerations
        # acc_base[:, 0] -> x_ddot, acc_base[:, 1] -> th1_ddot, acc_base[:, 2] -> th2_ddot
        acc_base = acc_base.clone()
        acc_base[:, 0:1] = acc_base[:, 0:1] - self.cart_damping * x_dot
        acc_base[:, 1:2] = acc_base[:, 1:2] - self.theta1_damping * th1_dot
        acc_base[:, 2 : 2 + 1] = acc_base[:, 2 : 2 + 1] - self.theta2_damping * th2_dot

        if self._residual is not None:
            accel_res = self._residual(torch.cat([obs_flat, action_flat], dim=-1))
            # Scalar gate based on features
            g = torch.sigmoid(self._residual_gate(phi))  # shape (B,1)
            acc = acc_base + g * accel_res
        else:
            acc = acc_base
        return acc  # shape: (B, 3)

    def _integrate(
        self,
        obs_flat: torch.Tensor,
        acc: torch.Tensor,
    ) -> torch.Tensor:
        x, s1, s2, c1, c2, x_dot, th1_dot, th2_dot, _ = self._split_observation(
            obs_flat
        )

        th1 = self._angles_from_sin_cos(s1, c1)
        th2 = self._angles_from_sin_cos(s2, c2)

        x_ddot = acc[:, 0:1]
        th1_ddot = acc[:, 1:2]
        th2_ddot = acc[:, 2:3]

        # Semi-implicit (symplectic) Euler for better stability
        x_dot_next = x_dot + self.dt * x_ddot
        th1_dot_next = th1_dot + self.dt * th1_ddot
        th2_dot_next = th2_dot + self.dt * th2_ddot

        x_next = x + self.dt * x_dot_next
        th1_next = th1 + self.dt * th1_dot_next
        th2_next = th2 + self.dt * th2_dot_next

        s1_next = torch.sin(th1_next)
        c1_next = torch.cos(th1_next)
        s2_next = torch.sin(th2_next)
        c2_next = torch.cos(th2_next)

        # Assemble next observation without constraint forces
        next_obs_wo_cfrc = torch.cat(
            [
                x_next,
                s1_next,
                s2_next,
                c1_next,
                c2_next,
                x_dot_next,
                th1_dot_next,
                th2_dot_next,
            ],
            dim=-1,
        )
        return next_obs_wo_cfrc

    def _predict_cfrc(self, phi: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
        return self._cfrc_head(torch.cat([phi, acc], dim=-1))

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
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

        leading_shape = observation.shape[:-1]
        obs_flat = observation.reshape(-1, self.obs_dim)
        act_flat = action.reshape(-1, self.action_dim)

        # Build features once for heads that need them
        x, s1, s2, c1, c2, x_dot, th1_dot, th2_dot, _ = self._split_observation(
            obs_flat
        )
        phi = self._build_phi(x, s1, s2, c1, c2, x_dot, th1_dot, th2_dot, act_flat)

        acc = self._predict_accelerations(obs_flat, act_flat)
        next_obs_wo_cfrc = self._integrate(obs_flat, acc)
        cfrc_next = self._predict_cfrc(phi, acc)

        next_obs = torch.cat([next_obs_wo_cfrc, cfrc_next], dim=-1)
        next_obs = next_obs.reshape(*leading_shape, self.obs_dim)
        return next_obs

    def compute_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        target_next_observation: torch.Tensor,
        loss_reduction: str = "mean",
        cfrc_weight: float = 0.1,
    ) -> torch.Tensor:
        """
        Standard MSE loss to the target next observation with a lower weight for
        constraint forces to reduce their dominance.
        """
        pred = self.forward(observation, action)
        if pred.shape != target_next_observation.shape:
            raise ValueError(
                f"Target shape {target_next_observation.shape} must match predictions {pred.shape}."
            )
        # Down-weight cfrc components (last 3 dims)
        err = pred - target_next_observation
        state_err = err[..., :8]
        cfrc_err = err[..., 8:]
        loss = state_err.pow(2).mean(dim=-1) + cfrc_weight * cfrc_err.pow(2).mean(
            dim=-1
        )
        if loss_reduction == "mean":
            return loss.mean()
        elif loss_reduction == "sum":
            return loss.sum()
        elif loss_reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown loss_reduction: {loss_reduction}")

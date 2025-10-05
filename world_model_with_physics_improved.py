import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModelPhysicsSymbolicImproved(nn.Module):
    """
    An improved physics-informed world model for InvertedDoublePendulum-v4.

    Key improvements over the original:
    1. Parameter constraints: Ensures physically valid parameters (positive masses, etc.)
    2. Analytical mass matrix: Closed-form equations instead of nested autograd
    3. Better numerical stability: Regularization and gradient clipping
    4. RK4 integration: More accurate than semi-implicit Euler
    5. Better initialization: Physics-informed parameter initialization

    State Representation:
        Observation vector [11]:
            [0] x: Cart position
            [1-4] Angles: [sin(θ1), sin(θ2_rel), cos(θ1), cos(θ2_rel)]
            [5-7] Velocities: [x_dot, θ1_dot, θ2_rel_dot]
            [8-10] Contact forces: [cfrc_1, cfrc_2, cfrc_3]
    """

    def __init__(
        self,
        obs_dim: int = 11,
        action_dim: int = 1,
        dt: float = 0.02,
        # Physical constants (will be constrained to positive values)
        mass_cart: float = 1.0,
        mass_link1: float = 0.1,
        mass_link2: float = 0.1,
        length_link1: float = 0.5,
        com_link1: float = 0.25,
        com_link2: float = 0.25,
        inertia_link1: float = 0.002,
        inertia_link2: float = 0.002,
        gravity: float = 9.81,
        cart_damping: float = 0.1,
        joint1_damping: float = 0.01,
        joint2_damping: float = 0.01,
        force_scale: float = 1.0,
        # Model improvements
        use_rk4: bool = True,
        mass_matrix_reg: float = 1e-4,
    ) -> None:
        super().__init__()

        if obs_dim != 11 or action_dim != 1:
            raise ValueError(
                "WorldModelPhysicsSymbolicImproved is configured for InvertedDoublePendulum-v4."
            )

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.dt = float(dt)
        self.use_rk4 = use_rk4
        self.mass_matrix_reg = mass_matrix_reg

        # Use log-space parameterization for positive parameters
        # actual_param = softplus(raw_param) ensures positivity
        # Initialize raw params such that softplus gives the desired initial value
        def inv_softplus(x):
            """Inverse of softplus for initialization"""
            return torch.log(torch.exp(torch.tensor(x)) - 1.0)

        # Physical parameters (log-space to ensure positivity)
        self.raw_m_c = nn.Parameter(inv_softplus(mass_cart))
        self.raw_m1 = nn.Parameter(inv_softplus(mass_link1))
        self.raw_m2 = nn.Parameter(inv_softplus(mass_link2))
        self.raw_l1 = nn.Parameter(inv_softplus(length_link1))
        self.raw_lc1 = nn.Parameter(inv_softplus(com_link1))
        self.raw_lc2 = nn.Parameter(inv_softplus(com_link2))
        self.raw_I1 = nn.Parameter(inv_softplus(inertia_link1))
        self.raw_I2 = nn.Parameter(inv_softplus(inertia_link2))
        self.raw_g = nn.Parameter(inv_softplus(gravity))
        self.raw_b_x = nn.Parameter(inv_softplus(cart_damping))
        self.raw_b1 = nn.Parameter(inv_softplus(joint1_damping))
        self.raw_b2 = nn.Parameter(inv_softplus(joint2_damping))
        self.raw_force_scale = nn.Parameter(inv_softplus(force_scale))

    def _get_positive_params(self):
        """Get actual positive parameters from raw (log-space) parameters."""
        return {
            "m_c": F.softplus(self.raw_m_c),
            "m1": F.softplus(self.raw_m1),
            "m2": F.softplus(self.raw_m2),
            "l1": F.softplus(self.raw_l1),
            "lc1": F.softplus(self.raw_lc1),
            "lc2": F.softplus(self.raw_lc2),
            "I1": F.softplus(self.raw_I1),
            "I2": F.softplus(self.raw_I2),
            "g": F.softplus(self.raw_g),
            "b_x": F.softplus(self.raw_b_x),
            "b1": F.softplus(self.raw_b1),
            "b2": F.softplus(self.raw_b2),
            "force_scale": F.softplus(self.raw_force_scale),
        }

    def get_learned_parameters(self) -> dict:
        """Returns current learned physical parameters."""
        params = self._get_positive_params()
        return {k: float(v.item()) for k, v in params.items()}

    def print_learned_parameters(self) -> None:
        """Prints current learned physical parameters."""
        params = self.get_learned_parameters()
        print("\nLearned Physical Parameters:")
        print("=" * 50)
        print(f"  Cart mass (m_c):       {params['m_c']:.6f} kg")
        print(f"  Link 1 mass (m1):      {params['m1']:.6f} kg")
        print(f"  Link 2 mass (m2):      {params['m2']:.6f} kg")
        print(f"  Link 1 length (l1):    {params['l1']:.6f} m")
        print(f"  Link 1 COM (lc1):      {params['lc1']:.6f} m")
        print(f"  Link 2 COM (lc2):      {params['lc2']:.6f} m")
        print(f"  Link 1 inertia (I1):   {params['I1']:.6f} kg⋅m²")
        print(f"  Link 2 inertia (I2):   {params['I2']:.6f} kg⋅m²")
        print(f"  Gravity (g):           {params['g']:.6f} m/s²")
        print(f"  Cart damping (b_x):    {params['b_x']:.6f} N⋅s/m")
        print(f"  Joint 1 damping (b1):  {params['b1']:.6f} N⋅m⋅s")
        print(f"  Joint 2 damping (b2):  {params['b2']:.6f} N⋅m⋅s")
        print(f"  Force scale:           {params['force_scale']:.6f}")
        print("=" * 50)

    @staticmethod
    def _split_observation(obs_flat: torch.Tensor):
        """Splits flattened observation tensor into individual components."""
        x = obs_flat[:, 0:1]
        sin_th1 = obs_flat[:, 1:2]
        sin_th2_rel = obs_flat[:, 2:3]
        cos_th1 = obs_flat[:, 3:4]
        cos_th2_rel = obs_flat[:, 4:5]
        x_dot = obs_flat[:, 5:6]
        th1_dot = obs_flat[:, 6:7]
        th2_rel_dot = obs_flat[:, 7:8]
        cfrc = obs_flat[:, 8:11]
        return (
            x,
            sin_th1,
            sin_th2_rel,
            cos_th1,
            cos_th2_rel,
            x_dot,
            th1_dot,
            th2_rel_dot,
            cfrc,
        )

    @staticmethod
    def _angles_from_sin_cos(s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Converts sine and cosine components back to angles using atan2."""
        return torch.atan2(s, c)

    def _compute_mass_matrix_analytical(
        self, th1: torch.Tensor, th2_rel: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """
        Compute mass matrix M(q) analytically using closed-form equations.
        This avoids expensive nested autograd and improves numerical stability.

        For a cart-double-pendulum system, the mass matrix is:
        M = [[M11, M12, M13],
             [M21, M22, M23],
             [M31, M32, M33]]

        Args:
            th1: First link angle [B, 1]
            th2_rel: Second link relative angle [B, 1]
            params: Dictionary of physical parameters

        Returns:
            Mass matrix [B, 3, 3]
        """
        m_c = params["m_c"]
        m1 = params["m1"]
        m2 = params["m2"]
        l1 = params["l1"]
        lc1 = params["lc1"]
        lc2 = params["lc2"]
        I1 = params["I1"]
        I2 = params["I2"]

        batch_size = th1.shape[0]
        th2_abs = th1 + th2_rel

        # Precompute trig functions
        # c1 = torch.cos(th1)  # Not needed in mass matrix
        s1 = torch.sin(th1)
        # c2 = torch.cos(th2_abs)  # Not needed in mass matrix
        s2 = torch.sin(th2_abs)

        # Mass matrix elements (derived from Lagrangian)
        # M11: x-x coupling
        M11 = m_c + m1 + m2

        # M12 = M21: x-theta1 coupling
        M12 = -(m1 * lc1 + m2 * l1) * s1 - m2 * lc2 * s2

        # M13 = M31: x-theta2 coupling
        M13 = -m2 * lc2 * s2

        # M22: theta1-theta1 coupling
        M22 = (
            (m1 * lc1**2 + m2 * l1**2 + I1)
            + m2 * lc2**2
            + I2
            + 2 * m2 * l1 * lc2 * torch.cos(th2_rel)
        )

        # M23 = M32: theta1-theta2 coupling
        M23 = m2 * lc2**2 + I2 + m2 * l1 * lc2 * torch.cos(th2_rel)

        # M33: theta2-theta2 coupling
        M33 = m2 * lc2**2 + I2

        # Construct symmetric mass matrix [B, 3, 3]
        M = torch.zeros(batch_size, 3, 3, device=th1.device, dtype=th1.dtype)
        M[:, 0, 0] = M11.squeeze(-1)
        M[:, 0, 1] = M12.squeeze(-1)
        M[:, 1, 0] = M12.squeeze(-1)
        M[:, 0, 2] = M13.squeeze(-1)
        M[:, 2, 0] = M13.squeeze(-1)
        M[:, 1, 1] = M22.squeeze(-1)
        M[:, 1, 2] = M23.squeeze(-1)
        M[:, 2, 1] = M23.squeeze(-1)
        M[:, 2, 2] = M33.squeeze(-1)

        # Add regularization for numerical stability
        reg = self.mass_matrix_reg * torch.eye(
            3, device=th1.device, dtype=th1.dtype
        ).unsqueeze(0)
        M = M + reg

        return M

    def _compute_coriolis_gravity_analytical(
        self, q: torch.Tensor, qdot: torch.Tensor, params: dict
    ) -> torch.Tensor:
        """
        Compute Coriolis, centrifugal, and gravity terms analytically.
        h = C(q, qdot) * qdot + G(q)

        Args:
            q: Generalized coordinates [B, 3] = [x, θ1, θ2_rel]
            qdot: Generalized velocities [B, 3]
            params: Dictionary of physical parameters

        Returns:
            Combined forces h [B, 3]
        """
        m1 = params["m1"]
        m2 = params["m2"]
        l1 = params["l1"]
        lc1 = params["lc1"]
        lc2 = params["lc2"]
        g = params["g"]

        th1 = q[:, 1:2]
        th2_rel = q[:, 2:3]
        th2_abs = th1 + th2_rel

        th1dot = qdot[:, 1:2]
        th2dot = qdot[:, 2:3]
        th2_absdot = th1dot + th2dot

        # Trig functions
        # s1 = torch.sin(th1)  # Not needed in h computation
        c1 = torch.cos(th1)
        # s2 = torch.sin(th2_abs)  # Not needed in h computation
        c2 = torch.cos(th2_abs)
        s_rel = torch.sin(th2_rel)

        # Coriolis/centrifugal terms (from d/dt(dL/dq) - dL/dq)
        # h_x: Forces on cart due to pendulum motion
        h_x = -(m1 * lc1 + m2 * l1) * c1 * th1dot**2 - m2 * lc2 * c2 * th2_absdot**2

        # h_th1: Torques on first joint
        h_th1 = (
            -m2 * l1 * lc2 * s_rel * th2dot * (2 * th1dot + th2dot)
            - (m1 * lc1 + m2 * l1) * g * c1
            - m2 * g * lc2 * c2
        )

        # h_th2: Torques on second joint
        h_th2 = m2 * l1 * lc2 * s_rel * th1dot**2 - m2 * g * lc2 * c2

        h = torch.cat([h_x, h_th1, h_th2], dim=-1)
        return h

    def _compute_qdd(
        self, q: torch.Tensor, qdot: torch.Tensor, force: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes generalized accelerations using analytical equations.
        Solves: M(q) qdd + h(q, qdot) = Q

        Args:
            q: Generalized coordinates [B, 3]
            qdot: Generalized velocities [B, 3]
            force: Applied cart force [B, 1]

        Returns:
            Generalized accelerations qdd [B, 3]
        """
        params = self._get_positive_params()

        th1 = q[:, 1:2]
        th2_rel = q[:, 2:3]

        # Compute mass matrix and dynamics terms analytically
        M = self._compute_mass_matrix_analytical(th1, th2_rel, params)
        h = self._compute_coriolis_gravity_analytical(q, qdot, params)

        # Generalized forces Q (applied force + damping)
        Q = torch.zeros_like(q)
        Q[:, 0:1] = force - params["b_x"] * qdot[:, 0:1]
        Q[:, 1:2] = -params["b1"] * qdot[:, 1:2]
        Q[:, 2:3] = -params["b2"] * qdot[:, 2:3]

        # Solve M qdd = Q - h for accelerations
        rhs = Q - h
        qdd = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)

        return qdd

    def _integrate_rk4(
        self, q: torch.Tensor, qdot: torch.Tensor, force: torch.Tensor
    ) -> tuple:
        """
        RK4 integration for more accurate state prediction.

        Args:
            q: Current generalized coordinates [B, 3]
            qdot: Current generalized velocities [B, 3]
            force: Applied force [B, 1]

        Returns:
            Tuple of (q_next, qdot_next)
        """
        dt = self.dt

        # RK4 stage 1
        k1_qdot = qdot
        k1_qdd = self._compute_qdd(q, qdot, force)

        # RK4 stage 2
        q2 = q + 0.5 * dt * k1_qdot
        qdot2 = qdot + 0.5 * dt * k1_qdd
        k2_qdot = qdot2
        k2_qdd = self._compute_qdd(q2, qdot2, force)

        # RK4 stage 3
        q3 = q + 0.5 * dt * k2_qdot
        qdot3 = qdot + 0.5 * dt * k2_qdd
        k3_qdot = qdot3
        k3_qdd = self._compute_qdd(q3, qdot3, force)

        # RK4 stage 4
        q4 = q + dt * k3_qdot
        qdot4 = qdot + dt * k3_qdd
        k4_qdot = qdot4
        k4_qdd = self._compute_qdd(q4, qdot4, force)

        # Combine stages
        q_next = q + (dt / 6.0) * (k1_qdot + 2 * k2_qdot + 2 * k3_qdot + k4_qdot)
        qdot_next = qdot + (dt / 6.0) * (k1_qdd + 2 * k2_qdd + 2 * k3_qdd + k4_qdd)

        return q_next, qdot_next

    def _integrate_euler(
        self, q: torch.Tensor, qdot: torch.Tensor, force: torch.Tensor
    ) -> tuple:
        """
        Semi-implicit Euler integration (faster but less accurate).

        Args:
            q: Current generalized coordinates [B, 3]
            qdot: Current generalized velocities [B, 3]
            force: Applied force [B, 1]

        Returns:
            Tuple of (q_next, qdot_next)
        """
        qdd = self._compute_qdd(q, qdot, force)

        # Semi-implicit Euler
        qdot_next = qdot + self.dt * qdd
        q_next = q + self.dt * qdot_next

        return q_next, qdot_next

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predicts next observation given current observation and action.

        Args:
            observation: Current observation [..., 11]
            action: Applied force [..., 1]

        Returns:
            Predicted next observation [..., 11]
        """
        # Input validation
        if observation.ndim < 2 or action.ndim < 2:
            raise ValueError("Both observation and action must be at least 2D tensors.")
        if observation.shape[:-1] != action.shape[:-1]:
            raise ValueError("Leading dimensions must match.")
        if observation.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected observation dim {self.obs_dim}, got {observation.shape[-1]}."
            )
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dim {self.action_dim}, got {action.shape[-1]}."
            )

        # Flatten to batch dimension
        leading_shape = observation.shape[:-1]
        obs_flat = observation.reshape(-1, self.obs_dim)
        act_flat = action.reshape(-1, self.action_dim)

        # Extract state variables
        x, s1, s2_rel, c1, c2_rel, x_dot, th1_dot, th2_rel_dot, _ = (
            self._split_observation(obs_flat)
        )
        th1 = self._angles_from_sin_cos(s1, c1)
        th2_rel = self._angles_from_sin_cos(s2_rel, c2_rel)

        # Build generalized coordinates and velocities
        q = torch.cat([x, th1, th2_rel], dim=-1)
        qdot = torch.cat([x_dot, th1_dot, th2_rel_dot], dim=-1)

        # Scale action to force
        params = self._get_positive_params()
        force = params["force_scale"] * act_flat

        # Integrate physics
        if self.use_rk4:
            q_next, qdot_next = self._integrate_rk4(q, qdot, force)
        else:
            q_next, qdot_next = self._integrate_euler(q, qdot, force)

        # Extract next state
        x_next = q_next[:, 0:1]
        th1_next = q_next[:, 1:2]
        th2_rel_next = q_next[:, 2:3]

        x_dot_next = qdot_next[:, 0:1]
        th1_dot_next = qdot_next[:, 1:2]
        th2_rel_dot_next = qdot_next[:, 2:3]

        # Convert angles to sin/cos
        s1_next = torch.sin(th1_next)
        c1_next = torch.cos(th1_next)
        s2_rel_next = torch.sin(th2_rel_next)
        c2_rel_next = torch.cos(th2_rel_next)

        # Physics prediction (without contact forces)
        next_obs_wo_cfrc = torch.cat(
            [
                x_next,
                s1_next,
                s2_rel_next,
                c1_next,
                c2_rel_next,
                x_dot_next,
                th1_dot_next,
                th2_rel_dot_next,
            ],
            dim=-1,
        )

        # Contact forces set to zero (not modeled)
        zeros_cfrc = torch.zeros(
            obs_flat.shape[0], 3, device=obs_flat.device, dtype=obs_flat.dtype
        )
        next_obs = torch.cat([next_obs_wo_cfrc, zeros_cfrc], dim=-1)

        # Reshape back to original dimensions
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
        Computes prediction error between model output and target next state.

        Args:
            observation: Current observation
            action: Applied action
            target_next_observation: True next observation
            loss_reduction: Reduction method ("mean", "sum", "none")
            cfrc_weight: Weight for contact force prediction error

        Returns:
            Scalar loss if reduction is "mean"/"sum", else per-sample losses
        """
        pred = self.forward(observation, action)
        if pred.shape != target_next_observation.shape:
            raise ValueError(
                f"Target shape {target_next_observation.shape} must match predictions {pred.shape}."
            )

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

import torch
import torch.nn as nn


class WorldModelPhysicsSymbolic(nn.Module):
    """
    A physics-informed world model for the InvertedDoublePendulum-v4 environment using Lagrangian mechanics.

    This model implements a learnable physics-based approach:
    - Uses exact physical equations derived from Lagrangian mechanics
    - All parameters (masses, lengths, inertias, damping, etc.) are LEARNABLE via gradient descent
    - Predicts next state using physical laws with learnable constants
    - Can fit true physical parameters from observed dynamics data

    State Representation:
        Observation vector [11]:
            [0] x: Cart position
            [1-4] Angles: [sin(θ1), sin(θ2_rel), cos(θ1), cos(θ2_rel)]
            [5-7] Velocities: [x_dot, θ1_dot, θ2_rel_dot]
            [8-10] Contact forces: [cfrc_1, cfrc_2, cfrc_3]

        Where:
            - θ2_rel: Relative angle of second link relative to first link
            - θ2_abs = θ1 + θ2_rel: Absolute angle of second link

    Control:
        - Single action: Horizontal force applied to cart (Newtons)
        - Force is scaled by force_scale parameter
    """

    def __init__(
        self,
        obs_dim: int = 11,
        action_dim: int = 1,
        dt: float = 0.02,
        # Physical constants (defaults are reasonable placeholders)
        mass_cart: float = 1.0,
        mass_link1: float = 0.1,
        mass_link2: float = 0.1,
        length_link1: float = 0.5,
        # length_link2 removed - not needed for dynamics (only lc2 matters)
        com_link1: float = 0.25,  # distance to COM from joint
        com_link2: float = 0.25,
        inertia_link1: float = 0.002,
        inertia_link2: float = 0.002,
        gravity: float = 9.81,
        # Linear damping (non-conservative generalized forces)
        cart_damping: float = 0.0,
        joint1_damping: float = 0.0,
        joint2_damping: float = 0.0,
        force_scale: float = 1.0,  # action to force scaling
    ) -> None:
        super().__init__()

        if obs_dim != 11 or action_dim != 1:
            raise ValueError(
                "WorldModelPhysicsSymbolic is configured for InvertedDoublePendulum-v4 with obs_dim=11 and action_dim=1."
            )

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.dt = float(dt)

        # Register physical constants as LEARNABLE parameters
        # These will be optimized via gradient descent during training
        self.m_c = nn.Parameter(torch.tensor(float(mass_cart)))
        self.m1 = nn.Parameter(torch.tensor(float(mass_link1)))
        self.m2 = nn.Parameter(torch.tensor(float(mass_link2)))
        self.l1 = nn.Parameter(torch.tensor(float(length_link1)))
        # l2 removed - not needed (only lc2 is used in physics equations)
        self.lc1 = nn.Parameter(torch.tensor(float(com_link1)))
        self.lc2 = nn.Parameter(torch.tensor(float(com_link2)))
        self.I1 = nn.Parameter(torch.tensor(float(inertia_link1)))
        self.I2 = nn.Parameter(torch.tensor(float(inertia_link2)))
        self.g = nn.Parameter(torch.tensor(float(gravity)))
        self.b_x = nn.Parameter(torch.tensor(float(cart_damping)))
        self.b1 = nn.Parameter(torch.tensor(float(joint1_damping)))
        self.b2 = nn.Parameter(torch.tensor(float(joint2_damping)))
        self.force_scale = nn.Parameter(torch.tensor(float(force_scale)))

    def get_learned_parameters(self) -> dict:
        """
        Returns a dictionary of the current learned physical parameters.

        Returns:
            Dictionary with parameter names and their current values as Python floats.
        """
        return {
            "mass_cart": float(self.m_c.item()),
            "mass_link1": float(self.m1.item()),
            "mass_link2": float(self.m2.item()),
            "length_link1": float(self.l1.item()),
            # length_link2 removed - not used in physics
            "com_link1": float(self.lc1.item()),
            "com_link2": float(self.lc2.item()),
            "inertia_link1": float(self.I1.item()),
            "inertia_link2": float(self.I2.item()),
            "gravity": float(self.g.item()),
            "cart_damping": float(self.b_x.item()),
            "joint1_damping": float(self.b1.item()),
            "joint2_damping": float(self.b2.item()),
            "force_scale": float(self.force_scale.item()),
        }

    def print_learned_parameters(self) -> None:
        """
        Prints the current learned physical parameters in a readable format.
        """
        params = self.get_learned_parameters()
        print("\nLearned Physical Parameters:")
        print("=" * 50)
        print(f"  Cart mass (m_c):       {params['mass_cart']:.6f} kg")
        print(f"  Link 1 mass (m1):      {params['mass_link1']:.6f} kg")
        print(f"  Link 2 mass (m2):      {params['mass_link2']:.6f} kg")
        print(f"  Link 1 length (l1):    {params['length_link1']:.6f} m")
        # Link 2 length (l2) removed - not used in physics calculations
        print(f"  Link 1 COM (lc1):      {params['com_link1']:.6f} m")
        print(f"  Link 2 COM (lc2):      {params['com_link2']:.6f} m")
        print(f"  Link 1 inertia (I1):   {params['inertia_link1']:.6f} kg⋅m²")
        print(f"  Link 2 inertia (I2):   {params['inertia_link2']:.6f} kg⋅m²")
        print(f"  Gravity (g):           {params['gravity']:.6f} m/s²")
        print(f"  Cart damping (b_x):    {params['cart_damping']:.6f} N⋅s/m")
        print(f"  Joint 1 damping (b1):  {params['joint1_damping']:.6f} N⋅m⋅s")
        print(f"  Joint 2 damping (b2):  {params['joint2_damping']:.6f} N⋅m⋅s")
        print(f"  Force scale:           {params['force_scale']:.6f}")
        print("=" * 50)

    @staticmethod
    def _split_observation(obs_flat: torch.Tensor):
        """
        Splits flattened observation tensor into individual components.

        Args:
            obs_flat: Batch of observations [B, 11]

        Returns:
            Tuple of tensors:
                - Cart position (x)
                - Link angles (sin/cos θ1, sin/cos θ2_rel)
                - Velocities (x_dot, θ1_dot, θ2_rel_dot)
                - Contact forces (cfrc)
        """
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
        """
        Converts sine and cosine components back to angles using atan2.

        Args:
            s: Sine component of angle
            c: Cosine component of angle

        Returns:
            Angle in radians in range [-π, π]
        """
        return torch.atan2(s, c)

    def _lagrangian(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """
        Computes the Lagrangian (L = T - V) for the double pendulum system.

        Implements the full Lagrangian mechanics for a cart with double pendulum:
        - Kinetic energy (T) includes cart, both links' COM motion and angular velocity
        - Potential energy (V) accounts for gravitational potential of both links

        Args:
            q: Generalized coordinates [B, 3] = [x, θ1, θ2_rel]
            qdot: Generalized velocities [B, 3] = [x_dot, θ1_dot, θ2_rel_dot]

        Returns:
            Lagrangian value for each batch element [B]
        """
        # Extract angles and velocities
        th1 = q[:, 1:2]
        th2_rel = q[:, 2:3]
        th2_abs = th1 + th2_rel

        xdot = qdot[:, 0:1]
        th1dot = qdot[:, 1:2]
        th2_reldot = qdot[:, 2:3]
        th2_absdot = th1dot + th2_reldot

        # Precompute trigonometric functions
        cos_th1 = torch.cos(th1)
        sin_th1 = torch.sin(th1)
        cos_th2_abs = torch.cos(th2_abs)
        sin_th2_abs = torch.sin(th2_abs)

        # Kinematics: COM linear velocities (planar, y up)
        x1dot = xdot + self.lc1 * cos_th1 * th1dot
        y1dot = -self.lc1 * sin_th1 * th1dot

        x2dot = xdot + self.l1 * cos_th1 * th1dot + self.lc2 * cos_th2_abs * th2_absdot
        y2dot = -self.l1 * sin_th1 * th1dot - self.lc2 * sin_th2_abs * th2_absdot

        # Kinetic energy
        T_cart = 0.5 * self.m_c * xdot.pow(2)
        T1 = 0.5 * self.m1 * (x1dot.pow(2) + y1dot.pow(2)) + 0.5 * self.I1 * th1dot.pow(
            2
        )
        T2 = 0.5 * self.m2 * (
            x2dot.pow(2) + y2dot.pow(2)
        ) + 0.5 * self.I2 * th2_absdot.pow(2)
        T = T_cart + T1 + T2

        # Potential energy (y up)
        y1 = self.lc1 * cos_th1
        y2 = self.l1 * cos_th1 + self.lc2 * cos_th2_abs
        V = self.m1 * self.g * y1 + self.m2 * self.g * y2

        L = T - V
        return L.squeeze(-1)  # [B, 1] -> [B]

    def _compute_qdd(
        self, q: torch.Tensor, qdot: torch.Tensor, force: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes generalized accelerations using the Euler-Lagrange equations.

        Solves the equation of motion:
            M(q) qdd + h(q, qdot) = Q
        where:
            - M(q): Mass/inertia matrix
            - h: Coriolis, centrifugal, and gravity terms
            - Q: Applied forces (cart force + damping)

        Uses automatic differentiation to compute:
            - Mass matrix M = ∂²L/∂qdot²
            - h = d/dq(∂L/∂qdot) qdot - ∂L/∂q

        Args:
            q: Generalized coordinates [B, 3]
            qdot: Generalized velocities [B, 3]
            force: Applied cart force [B, 1]

        Returns:
            Generalized accelerations qdd [B, 3]
        """
        # Enable gradients for physics computations even if in no_grad context
        # (needed for validation/evaluation when model.eval() and torch.no_grad() are active)
        with torch.enable_grad():
            # Ensure q, qdot require gradients for autograd-based Lagrangian calculus
            q_req = q.detach().requires_grad_(True)
            qdot_req = qdot.detach().requires_grad_(True)

            L = self._lagrangian(q_req, qdot_req)  # [B]

            # Compute gradients of Lagrangian
            dLdq = torch.autograd.grad(L.sum(), q_req, create_graph=True)[0]  # [B, 3]
            dLdqdot = torch.autograd.grad(L.sum(), qdot_req, create_graph=True)[
                0
            ]  # [B, 3]

            # Mass matrix M_ij = ∂/∂qdot_j (∂L/∂qdot_i)
            M_rows = []
            for i in range(3):
                grad_i = torch.autograd.grad(
                    dLdqdot[:, i].sum(), qdot_req, retain_graph=True, create_graph=True
                )[0]
                M_rows.append(grad_i.unsqueeze(1))  # [B, 1, 3]
            M = torch.cat(M_rows, dim=1)  # [B, 3, 3]

            # A_ij = ∂/∂q_j (∂L/∂qdot_i)
            A_rows = []
            for i in range(3):
                grad_i = torch.autograd.grad(
                    dLdqdot[:, i].sum(), q_req, retain_graph=True, create_graph=True
                )[0]
                A_rows.append(grad_i.unsqueeze(1))  # [B, 1, 3]
            A = torch.cat(A_rows, dim=1)  # [B, 3, 3]

            # Coriolis, centrifugal, and gravity terms: h = A(q,qdot) @ qdot - dLdq
            h = torch.bmm(A, qdot_req.unsqueeze(-1)).squeeze(-1) - dLdq  # [B, 3]

        # Generalized forces Q: applied force on cart DOF + viscous damping
        Q = torch.zeros_like(q)  # [B, 3]
        Q[:, 0:1] = force - self.b_x * qdot[:, 0:1]  # Cart: applied force - damping
        Q[:, 1:2] = -self.b1 * qdot[:, 1:2]  # Joint 1: damping only
        Q[:, 2:3] = -self.b2 * qdot[:, 2:3]  # Joint 2: damping only

        # Solve M qdd = Q - h for accelerations
        rhs = Q - h  # [B, 3]
        qdd = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)  # [B, 3]
        return qdd

    def _integrate(self, obs_flat: torch.Tensor, qdd: torch.Tensor) -> torch.Tensor:
        """
        Performs semi-implicit Euler integration to compute next state.

        Integration scheme:
            1. Update velocities using accelerations (implicit)
            2. Update positions using new velocities (explicit)
            3. Convert angles back to sin/cos representation

        Args:
            obs_flat: Current flattened observation [B, 11]
            qdd: Computed accelerations [B, 3]

        Returns:
            Next observation (without contact forces) [B, 8]
        """
        x, s1, s2_rel, c1, c2_rel, x_dot, th1_dot, th2_rel_dot, _ = (
            self._split_observation(obs_flat)
        )

        th1 = self._angles_from_sin_cos(s1, c1)
        th2_rel = self._angles_from_sin_cos(s2_rel, c2_rel)

        x_ddot = qdd[:, 0:1]
        th1_ddot = qdd[:, 1:2]
        th2_rel_ddot = qdd[:, 2:3]

        # Semi-implicit Euler
        x_dot_next = x_dot + self.dt * x_ddot
        th1_dot_next = th1_dot + self.dt * th1_ddot
        th2_rel_dot_next = th2_rel_dot + self.dt * th2_rel_ddot

        x_next = x + self.dt * x_dot_next
        th1_next = th1 + self.dt * th1_dot_next
        th2_rel_next = th2_rel + self.dt * th2_rel_dot_next

        s1_next = torch.sin(th1_next)
        c1_next = torch.cos(th1_next)
        s2_rel_next = torch.sin(th2_rel_next)
        c2_rel_next = torch.cos(th2_rel_next)

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
        return next_obs_wo_cfrc

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predicts next observation given current observation and action.

        Prediction steps:
            1. Extract state variables from observation
            2. Convert sin/cos to angles
            3. Compute accelerations using Lagrangian mechanics
            4. Integrate state forward in time
            5. Set contact forces to zero (not modeled)

        Args:
            observation: Current observation [..., 11]
            action: Applied force [..., 1]

        Returns:
            Predicted next observation [..., 11]
        """
        # Input validation
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

        # Flatten to batch dimension for processing
        leading_shape = observation.shape[:-1]
        obs_flat = observation.reshape(-1, self.obs_dim)
        act_flat = action.reshape(-1, self.action_dim)

        # Extract and reconstruct state variables
        x, s1, s2_rel, c1, c2_rel, x_dot, th1_dot, th2_rel_dot, _ = (
            self._split_observation(obs_flat)
        )
        th1 = self._angles_from_sin_cos(s1, c1)
        th2_rel = self._angles_from_sin_cos(s2_rel, c2_rel)

        # Build generalized coordinates and velocities
        q = torch.cat([x, th1, th2_rel], dim=-1)  # [B, 3]
        qdot = torch.cat([x_dot, th1_dot, th2_rel_dot], dim=-1)  # [B, 3]

        # Scale action to force
        force = self.force_scale * act_flat  # [B, 1]

        # Compute accelerations and integrate
        qdd = self._compute_qdd(q, qdot, force)
        next_obs_wo_cfrc = self._integrate(obs_flat, qdd)

        # Contact forces are not modeled symbolically → set to zero
        zeros_cfrc = torch.zeros(
            obs_flat.shape[0], 3, device=obs_flat.device, dtype=obs_flat.dtype
        )
        next_obs = torch.cat([next_obs_wo_cfrc, zeros_cfrc], dim=-1)

        # Reshape back to original leading dimensions
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

        Loss components:
            - Main state error: MSE over positions and velocities
            - Contact force error: Weighted MSE over contact forces

        Args:
            observation: Current observation
            action: Applied action
            target_next_observation: True next observation
            loss_reduction: Reduction method for batch ("mean", "sum", "none")
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

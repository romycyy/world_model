#!/usr/bin/env python3
"""
Symbolic physics-based world model for the 2-link planar reacher environment.

This model uses only physical parameters (no neural network components) to predict
the next state given the current state and action. All parameters are learnable
from data.

State Representation:
    Observation vector [10]:
        [0] cos(θ1): Cosine of first joint angle
        [1] cos(θ2): Cosine of second joint angle
        [2] sin(θ1): Sine of first joint angle
        [3] sin(θ2): Sine of second joint angle
        [4] target_x: X-coordinate of target (constant during episode)
        [5] target_y: Y-coordinate of target (constant during episode)
        [6] θ1_dot: Angular velocity of first joint
        [7] θ2_dot: Angular velocity of second joint
        [8] to_target_x: Fingertip to target distance in x
        [9] to_target_y: Fingertip to target distance in y

    Action vector [2]:
        [0] τ1: Torque applied to first joint (shoulder)
        [1] τ2: Torque applied to second joint (elbow/wrist)

Physical Model:
    - 2-link planar manipulator in horizontal plane
    - Revolute joints with damping
    - Lagrangian mechanics: M(q)q̈ + C(q,q̇)q̇ + G(q) = τ
    - For horizontal plane: gravity term G(q) ≈ 0 or very small

Learnable Parameters:
    - Link masses: m1, m2
    - Link lengths: l1, l2
    - Center of mass positions: lc1, lc2
    - Link inertias: I1, I2
    - Joint damping coefficients: b1, b2
    - Torque scaling factors: torque_scale_1, torque_scale_2
    - Optional: small gravity component if not perfectly horizontal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ReacherPhysicsModel(nn.Module):
    """
    Symbolic physics-based world model for 2-link planar reacher.

    Uses analytical equations derived from Lagrangian mechanics to predict
    the next state given current state and action. All physical parameters
    are learnable from data.
    """

    def __init__(
        self,
        obs_dim: int = 10,
        action_dim: int = 2,
        dt: float = 0.02,
        # Physical parameters (initial values)
        mass_link1: float = 0.1,
        mass_link2: float = 0.1,
        length_link1: float = 0.1,
        length_link2: float = 0.1,
        com_link1: float = 0.05,
        com_link2: float = 0.05,
        inertia_link1: float = 0.001,
        inertia_link2: float = 0.001,
        joint1_damping: float = 0.01,
        joint2_damping: float = 0.01,
        torque_scale_1: float = 1.0,
        torque_scale_2: float = 1.0,
        gravity: float = 0.0,  # Small value if not perfectly horizontal
        # Model configuration
        use_rk4: bool = True,
        mass_matrix_reg: float = 1e-6,
    ) -> None:
        """
        Initialize the reacher physics model.

        Args:
            obs_dim: Observation dimension (should be 10 or 11; extra dims passed through)
            action_dim: Action dimension (should be 2)
            dt: Integration timestep (seconds)
            mass_link1: Mass of first link (kg)
            mass_link2: Mass of second link (kg)
            length_link1: Length of first link (m)
            length_link2: Length of second link (m)
            com_link1: Center of mass of first link from joint (m)
            com_link2: Center of mass of second link from joint (m)
            inertia_link1: Moment of inertia of first link (kg⋅m²)
            inertia_link2: Moment of inertia of second link (kg⋅m²)
            joint1_damping: Damping coefficient for first joint (N⋅m⋅s)
            joint2_damping: Damping coefficient for second joint (N⋅m⋅s)
            torque_scale_1: Scaling factor for first joint torque
            torque_scale_2: Scaling factor for second joint torque
            gravity: Gravity component (m/s²), usually 0 for horizontal plane
            use_rk4: Whether to use RK4 integration (more accurate)
            mass_matrix_reg: Regularization for mass matrix inversion
        """
        super().__init__()

        if obs_dim < 10:
            raise ValueError(f"Reacher model expects obs_dim >= 10, got {obs_dim}")
        if action_dim != 2:
            raise ValueError(f"Reacher model expects action_dim=2, got {action_dim}")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.dt = dt
        self.use_rk4 = use_rk4
        self.mass_matrix_reg = mass_matrix_reg

        # Use softplus parameterization to ensure positive values
        def inv_softplus(x):
            """Inverse softplus for initialization"""
            return torch.log(torch.exp(torch.tensor(x, dtype=torch.float32)) - 1.0)

        def inv_sigmoid(x):
            """Inverse sigmoid for initialization of bounded parameters"""
            x = torch.clamp(torch.tensor(x, dtype=torch.float32), 1e-6, 1 - 1e-6)
            return torch.log(x / (1 - x))

        # ========== LEARNABLE PHYSICAL PARAMETERS WITH CONSTRAINTS ==========
        #
        # We use a physically-informed parameterization to ensure all learned
        # parameters satisfy rigid body mechanics constraints.
        #
        # Mass and Length: Must be positive (softplus)
        self.raw_m1 = nn.Parameter(inv_softplus(mass_link1))
        self.raw_m2 = nn.Parameter(inv_softplus(mass_link2))
        self.raw_l1 = nn.Parameter(inv_softplus(length_link1))
        self.raw_l2 = nn.Parameter(inv_softplus(length_link2))

        # Center of Mass: Must be in [0, length]
        # Parameterize as: lc = sigmoid(raw_lc_ratio) * length
        # where raw_lc_ratio determines position along link
        init_lc1_ratio = com_link1 / length_link1 if length_link1 > 0 else 0.5
        init_lc2_ratio = com_link2 / length_link2 if length_link2 > 0 else 0.5
        self.raw_lc1_ratio = nn.Parameter(inv_sigmoid(init_lc1_ratio))
        self.raw_lc2_ratio = nn.Parameter(inv_sigmoid(init_lc2_ratio))

        # Moment of Inertia: Must satisfy parallel axis theorem
        # For a rigid body rotating about one end:
        #   I_end = I_com + m * lc²
        # where I_com is inertia about center of mass.
        #
        # For a rod-like body: 0 ≤ I_com ≤ (1/12) * m * L²
        #   - I_com = 0: point mass at COM
        #   - I_com = (1/12)*m*L²: uniform rod
        #
        # We parameterize as: I_com = beta * (1/12) * m * L²
        # where beta ∈ [0, 1] is learned via sigmoid
        #
        # This ensures: I_min = m*lc² (point mass) ≤ I ≤ (1/12)*m*L² + m*lc² (uniform rod)

        # Initial beta values: try to match given inertia
        # beta = (I - m*lc²) / ((1/12)*m*L² - m*lc²) if possible, else 0.5
        def compute_init_beta(I_init, mass, length, com):
            I_com_init = I_init - mass * com**2
            I_com_max = (1 / 12) * mass * length**2
            beta_val = I_com_init / I_com_max if I_com_max > 1e-9 else 0.5
            return float(torch.clamp(torch.tensor(beta_val), 0.0, 1.0))

        init_beta1 = compute_init_beta(
            inertia_link1, mass_link1, length_link1, com_link1
        )
        init_beta2 = compute_init_beta(
            inertia_link2, mass_link2, length_link2, com_link2
        )

        self.raw_I1_beta = nn.Parameter(inv_sigmoid(init_beta1))
        self.raw_I2_beta = nn.Parameter(inv_sigmoid(init_beta2))

        # Damping coefficients: Must be positive, reasonable upper bound
        # Typical damping: 0.001 to 1.0 N⋅m⋅s
        self.raw_b1 = nn.Parameter(inv_softplus(joint1_damping))
        self.raw_b2 = nn.Parameter(inv_softplus(joint2_damping))

        # Torque scaling: Must be positive, reasonable range [0.1, 10.0]
        # Parameterize as: torque_scale = 0.1 + sigmoid(raw) * 9.9
        init_torque1_ratio = (torque_scale_1 - 0.1) / 9.9
        init_torque2_ratio = (torque_scale_2 - 0.1) / 9.9
        self.raw_torque_scale_1 = nn.Parameter(
            inv_sigmoid(torch.clamp(torch.tensor(init_torque1_ratio), 0.01, 0.99))
        )
        self.raw_torque_scale_2 = nn.Parameter(
            inv_sigmoid(torch.clamp(torch.tensor(init_torque2_ratio), 0.01, 0.99))
        )

        # Gravity: Small value for horizontal plane [0, 10.0] m/s²
        # Parameterize as: g = sigmoid(raw_g) * 10.0
        init_g_ratio = gravity / 10.0 if gravity < 10.0 else 0.001
        self.raw_g = nn.Parameter(
            inv_sigmoid(torch.clamp(torch.tensor(init_g_ratio), 1e-6, 1 - 1e-6))
        )

        # Physical bounds on parameters (for additional safety)
        # Add small epsilon to lower bounds to avoid floating point issues
        eps = 1e-7
        self.param_bounds = {
            "m1": (0.01 - eps, 2.0),  # Mass: 10g to 2kg
            "m2": (0.01 - eps, 2.0),
            "l1": (0.01 - eps, 0.5),  # Length: 1cm to 50cm
            "l2": (0.01 - eps, 0.5),
            "b1": (1e-4 - eps, 1.0),  # Damping: 0.0001 to 1.0 N⋅m⋅s
            "b2": (1e-4 - eps, 1.0),
        }

    def _get_positive_params(self) -> Dict[str, torch.Tensor]:
        """
        Get actual physical parameters with all constraints applied.

        This method enforces:
        1. Positivity constraints (mass, length, damping)
        2. Geometric constraints (COM within link)
        3. Inertia consistency (parallel axis theorem)
        4. Reasonable physical bounds on all parameters
        """
        # Step 1: Get basic positive parameters
        m1 = torch.clamp(F.softplus(self.raw_m1), *self.param_bounds["m1"])
        m2 = torch.clamp(F.softplus(self.raw_m2), *self.param_bounds["m2"])
        l1 = torch.clamp(F.softplus(self.raw_l1), *self.param_bounds["l1"])
        l2 = torch.clamp(F.softplus(self.raw_l2), *self.param_bounds["l2"])

        # Step 2: COM positions - constrained to be within [0, length]
        lc1_ratio = torch.sigmoid(self.raw_lc1_ratio)  # ∈ [0, 1]
        lc2_ratio = torch.sigmoid(self.raw_lc2_ratio)  # ∈ [0, 1]
        lc1 = lc1_ratio * l1  # ∈ [0, l1]
        lc2 = lc2_ratio * l2  # ∈ [0, l2]

        # Step 3: Inertia - computed from parallel axis theorem
        # I_end = I_com + m * lc²
        # where I_com = beta * (1/12) * m * L²
        beta1 = torch.sigmoid(self.raw_I1_beta)  # ∈ [0, 1]
        beta2 = torch.sigmoid(self.raw_I2_beta)  # ∈ [0, 1]

        I1_com = beta1 * (1 / 12) * m1 * l1 * l1
        I2_com = beta2 * (1 / 12) * m2 * l2 * l2

        I1 = I1_com + m1 * lc1 * lc1  # Parallel axis theorem
        I2 = I2_com + m2 * lc2 * lc2

        # Step 4: Damping coefficients
        b1 = torch.clamp(F.softplus(self.raw_b1), *self.param_bounds["b1"])
        b2 = torch.clamp(F.softplus(self.raw_b2), *self.param_bounds["b2"])

        # Step 5: Torque scaling - bounded to [0.1, 10.0]
        torque_scale_1 = 0.1 + torch.sigmoid(self.raw_torque_scale_1) * 9.9
        torque_scale_2 = 0.1 + torch.sigmoid(self.raw_torque_scale_2) * 9.9

        # Step 6: Gravity - bounded to [0, 10.0] m/s²
        g = torch.sigmoid(self.raw_g) * 10.0

        return {
            "m1": m1,
            "m2": m2,
            "l1": l1,
            "l2": l2,
            "lc1": lc1,
            "lc2": lc2,
            "I1": I1,
            "I2": I2,
            "b1": b1,
            "b2": b2,
            "torque_scale_1": torque_scale_1,
            "torque_scale_2": torque_scale_2,
            "g": g,
        }

    def get_learned_parameters(self) -> Dict[str, float]:
        """Returns current learned physical parameters as a dictionary."""
        params = self._get_positive_params()
        return {k: float(v.item()) for k, v in params.items()}

    def print_learned_parameters(self) -> None:
        """Print current learned physical parameters in a readable format."""
        params = self.get_learned_parameters()

        # Compute derived quantities for physical interpretation
        m1, m2 = params["m1"], params["m2"]
        l1, l2 = params["l1"], params["l2"]
        lc1, lc2 = params["lc1"], params["lc2"]
        I1, I2 = params["I1"], params["I2"]

        # COM position ratios (0 = at joint, 1 = at end)
        lc1_ratio = lc1 / l1 if l1 > 0 else 0
        lc2_ratio = lc2 / l2 if l2 > 0 else 0

        # Inertia components (from parallel axis theorem)
        I1_com = I1 - m1 * lc1 * lc1
        I2_com = I2 - m2 * lc2 * lc2

        # Beta values (mass distribution: 0 = point mass, 1 = uniform rod)
        I1_com_max = (1 / 12) * m1 * l1 * l1
        I2_com_max = (1 / 12) * m2 * l2 * l2
        beta1 = I1_com / I1_com_max if I1_com_max > 1e-9 else 0
        beta2 = I2_com / I2_com_max if I2_com_max > 1e-9 else 0

        print("\n" + "=" * 70)
        print("LEARNED PHYSICAL PARAMETERS (WITH CONSTRAINTS)")
        print("=" * 70)

        print("\n📏 LINK PROPERTIES:")
        print(f"  Link 1: m = {m1:.6f} kg, L = {l1:.6f} m")
        print(f"  Link 2: m = {m2:.6f} kg, L = {l2:.6f} m")
        print(f"  Mass ratio (m1/m2): {m1 / m2:.3f}")

        print("\n📍 CENTER OF MASS (COM):")
        print(f"  Link 1: lc = {lc1:.6f} m ({lc1_ratio * 100:.1f}% along link)")
        print(f"  Link 2: lc = {lc2:.6f} m ({lc2_ratio * 100:.1f}% along link)")
        print("  ✓ Constraint satisfied: 0 ≤ lc ≤ L")

        print("\n🔄 MOMENT OF INERTIA (about joint):")
        print(f"  Link 1: I = {I1:.6f} kg⋅m²")
        print(f"    - I_com = {I1_com:.6f} kg⋅m² (about COM)")
        print(f"    - m*lc² = {m1 * lc1 * lc1:.6f} kg⋅m² (parallel axis term)")
        print(f"    - β₁ = {beta1:.3f} (0=point mass, 1=uniform rod)")
        print(f"  Link 2: I = {I2:.6f} kg⋅m²")
        print(f"    - I_com = {I2_com:.6f} kg⋅m² (about COM)")
        print(f"    - m*lc² = {m2 * lc2 * lc2:.6f} kg⋅m² (parallel axis term)")
        print(f"    - β₂ = {beta2:.3f} (0=point mass, 1=uniform rod)")
        print("  ✓ Parallel axis theorem: I = I_com + m*lc²")

        print("\n⚙️  ACTUATION & DAMPING:")
        print(f"  Joint 1 damping: b = {params['b1']:.6f} N⋅m⋅s")
        print(f"  Joint 2 damping: b = {params['b2']:.6f} N⋅m⋅s")
        print(f"  Joint 1 torque scale: {params['torque_scale_1']:.6f}")
        print(f"  Joint 2 torque scale: {params['torque_scale_2']:.6f}")

        print("\n🌍 GRAVITY:")
        print(f"  g = {params['g']:.6f} m/s² (horizontal plane ≈ 0)")

        print("\n✅ ALL PHYSICAL CONSTRAINTS SATISFIED:")
        print(f"  • Mass: {m1:.3f}, {m2:.3f} ∈ [0.01, 2.0] kg")
        print(f"  • Length: {l1:.3f}, {l2:.3f} ∈ [0.01, 0.5] m")
        print(f"  • COM within link: lc/L = {lc1_ratio:.3f}, {lc2_ratio:.3f} ∈ [0, 1]")
        print("  • Inertia physically consistent via parallel axis theorem")
        print(
            f"  • Damping: {params['b1']:.3f}, {params['b2']:.3f} ∈ [0.0001, 1.0] N⋅m⋅s"
        )
        print("  • Torque scale: ∈ [0.1, 10.0]")
        print("=" * 70)

    @staticmethod
    def _split_observation(obs_flat: torch.Tensor) -> Tuple:
        """Split flattened observation into individual components."""
        cos_th1 = obs_flat[:, 0:1]
        cos_th2 = obs_flat[:, 1:2]
        sin_th1 = obs_flat[:, 2:3]
        sin_th2 = obs_flat[:, 3:4]
        target_x = obs_flat[:, 4:5]
        target_y = obs_flat[:, 5:6]
        th1_dot = obs_flat[:, 6:7]
        th2_dot = obs_flat[:, 7:8]
        to_target_x = obs_flat[:, 8:9]
        to_target_y = obs_flat[:, 9:10]

        return (
            cos_th1,
            cos_th2,
            sin_th1,
            sin_th2,
            target_x,
            target_y,
            th1_dot,
            th2_dot,
            to_target_x,
            to_target_y,
        )

    @staticmethod
    def _angles_from_sin_cos(s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Recover angle from sine and cosine using atan2."""
        return torch.atan2(s, c)

    def _compute_mass_matrix(
        self, th1: torch.Tensor, th2: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the mass matrix M(q) for a 2-link planar manipulator.

        For a 2-link planar arm, the mass matrix is:
        M = [[M11, M12],
             [M21, M22]]

        where:
        M11 = I1 + I2 + m1*lc1² + m2*(l1² + lc2² + 2*l1*lc2*cos(θ2))
        M12 = M21 = I2 + m2*lc2² + m2*l1*lc2*cos(θ2)
        M22 = I2 + m2*lc2²

        Args:
            th1: First joint angle [B, 1]
            th2: Second joint angle [B, 1]
            params: Dictionary of physical parameters

        Returns:
            Mass matrix M [B, 2, 2]
        """
        m1 = params["m1"]
        m2 = params["m2"]
        l1 = params["l1"]
        lc1 = params["lc1"]
        lc2 = params["lc2"]
        I1 = params["I1"]
        I2 = params["I2"]

        batch_size = th1.shape[0]

        # Compute mass matrix elements
        cos_th2 = torch.cos(th2)

        M11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos_th2)
        M12 = I2 + m2 * lc2**2 + m2 * l1 * lc2 * cos_th2
        M22 = I2 + m2 * lc2**2

        # Construct symmetric mass matrix [B, 2, 2]
        M = torch.zeros(batch_size, 2, 2, device=th1.device, dtype=th1.dtype)
        M[:, 0, 0] = M11.squeeze(-1)
        M[:, 0, 1] = M12.squeeze(-1)
        M[:, 1, 0] = M12.squeeze(-1)  # Symmetric
        M[:, 1, 1] = M22.squeeze(-1)

        # Add regularization for numerical stability
        reg = self.mass_matrix_reg * torch.eye(
            2, device=th1.device, dtype=th1.dtype
        ).unsqueeze(0)
        M = M + reg

        return M

    def _compute_coriolis_gravity(
        self, q: torch.Tensor, qdot: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute Coriolis, centrifugal, and gravity terms.

        For a 2-link planar arm:
        h1 = -m2*l1*lc2*sin(θ2)*(2*θ1_dot*θ2_dot + θ2_dot²) + g*[m1*lc1*sin(θ1) + m2*(l1*sin(θ1) + lc2*sin(θ1+θ2))]
        h2 = m2*l1*lc2*sin(θ2)*θ1_dot² + g*m2*lc2*sin(θ1+θ2)

        Note: For horizontal plane, gravity component is typically 0 or very small

        Args:
            q: Generalized coordinates [B, 2] = [θ1, θ2]
            qdot: Generalized velocities [B, 2]
            params: Dictionary of physical parameters

        Returns:
            Combined force vector h [B, 2]
        """
        m1 = params["m1"]
        m2 = params["m2"]
        l1 = params["l1"]
        lc1 = params["lc1"]
        lc2 = params["lc2"]
        g = params["g"]

        th1 = q[:, 0:1]
        th2 = q[:, 1:2]
        th1_dot = qdot[:, 0:1]
        th2_dot = qdot[:, 1:2]

        sin_th1 = torch.sin(th1)
        sin_th2 = torch.sin(th2)
        sin_th12 = torch.sin(th1 + th2)

        # Coriolis and centrifugal terms
        h1_coriolis = -m2 * l1 * lc2 * sin_th2 * (2 * th1_dot * th2_dot + th2_dot**2)
        h2_coriolis = m2 * l1 * lc2 * sin_th2 * th1_dot**2

        # Gravity terms (typically 0 for horizontal plane)
        h1_gravity = g * (m1 * lc1 * sin_th1 + m2 * (l1 * sin_th1 + lc2 * sin_th12))
        h2_gravity = g * m2 * lc2 * sin_th12

        h1 = h1_coriolis + h1_gravity
        h2 = h2_coriolis + h2_gravity

        h = torch.cat([h1, h2], dim=-1)
        return h

    def _compute_qdd(
        self, q: torch.Tensor, qdot: torch.Tensor, torques: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute generalized accelerations from equation of motion.

        Solves: M(q)q̈ + C(q,q̇)q̇ + G(q) + D*q̇ = τ

        where D is the damping matrix.

        Args:
            q: Generalized coordinates [B, 2] = [θ1, θ2]
            qdot: Generalized velocities [B, 2]
            torques: Applied torques [B, 2]

        Returns:
            Generalized accelerations qdd [B, 2]
        """
        params = self._get_positive_params()

        th1 = q[:, 0:1]
        th2 = q[:, 1:2]

        # Compute dynamics terms
        M = self._compute_mass_matrix(th1, th2, params)
        h = self._compute_coriolis_gravity(q, qdot, params)

        # Damping forces
        damping = torch.stack(
            [params["b1"] * qdot[:, 0], params["b2"] * qdot[:, 1]], dim=-1
        )

        # Equation of motion: M*qdd = τ - h - damping
        rhs = torques - h - damping

        # Solve for accelerations
        qdd = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)

        return qdd

    def _compute_fingertip_position(
        self, th1: torch.Tensor, th2: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the (x, y) position of the fingertip in world coordinates.

        For a 2-link arm:
        fingertip_x = l1*cos(θ1) + l2*cos(θ1 + θ2)
        fingertip_y = l1*sin(θ1) + l2*sin(θ1 + θ2)

        Args:
            th1: First joint angle [B, 1]
            th2: Second joint angle [B, 1]
            params: Dictionary of physical parameters

        Returns:
            Tuple of (fingertip_x, fingertip_y) each [B, 1]
        """
        l1 = params["l1"]
        l2 = params["l2"]

        # Forward kinematics
        fingertip_x = l1 * torch.cos(th1) + l2 * torch.cos(th1 + th2)
        fingertip_y = l1 * torch.sin(th1) + l2 * torch.sin(th1 + th2)

        return fingertip_x, fingertip_y

    def _integrate_rk4(
        self, q: torch.Tensor, qdot: torch.Tensor, torques: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fourth-order Runge-Kutta integration for state evolution.

        Args:
            q: Current angles [B, 2]
            qdot: Current angular velocities [B, 2]
            torques: Applied torques [B, 2]

        Returns:
            Tuple of (q_next, qdot_next)
        """
        dt = self.dt

        # RK4 stage 1
        k1_qdot = qdot
        k1_qdd = self._compute_qdd(q, qdot, torques)

        # RK4 stage 2
        q2 = q + 0.5 * dt * k1_qdot
        qdot2 = qdot + 0.5 * dt * k1_qdd
        k2_qdot = qdot2
        k2_qdd = self._compute_qdd(q2, qdot2, torques)

        # RK4 stage 3
        q3 = q + 0.5 * dt * k2_qdot
        qdot3 = qdot + 0.5 * dt * k2_qdd
        k3_qdot = qdot3
        k3_qdd = self._compute_qdd(q3, qdot3, torques)

        # RK4 stage 4
        q4 = q + dt * k3_qdot
        qdot4 = qdot + dt * k3_qdd
        k4_qdot = qdot4
        k4_qdd = self._compute_qdd(q4, qdot4, torques)

        # Combine stages
        q_next = q + (dt / 6.0) * (k1_qdot + 2 * k2_qdot + 2 * k3_qdot + k4_qdot)
        qdot_next = qdot + (dt / 6.0) * (k1_qdd + 2 * k2_qdd + 2 * k3_qdd + k4_qdd)

        return q_next, qdot_next

    def _integrate_euler(
        self, q: torch.Tensor, qdot: torch.Tensor, torques: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Semi-implicit Euler integration (faster but less accurate).

        Args:
            q: Current angles [B, 2]
            qdot: Current angular velocities [B, 2]
            torques: Applied torques [B, 2]

        Returns:
            Tuple of (q_next, qdot_next)
        """
        qdd = self._compute_qdd(q, qdot, torques)

        # Semi-implicit Euler
        qdot_next = qdot + self.dt * qdd
        q_next = q + self.dt * qdot_next

        return q_next, qdot_next

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next observation given current observation and action.

        Args:
            observation: Current observation [B, obs_dim] (at least 10 dims)
            action: Applied torques [B, 2]

        Returns:
            Predicted next observation [B, obs_dim]
        """
        # Input validation
        if observation.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected obs_dim={self.obs_dim}, got {observation.shape[-1]}"
            )
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got {action.shape[-1]}"
            )

        # Flatten to batch dimension
        obs_flat = observation.reshape(-1, self.obs_dim)
        act_flat = action.reshape(-1, self.action_dim)

        # Extract the first 10 dimensions for physics computation
        obs_physics = obs_flat[:, :10]

        # If there are extra dimensions (e.g., dim 10+), save them to pass through
        extra_dims = obs_flat[:, 10:] if self.obs_dim > 10 else None

        # Extract state components (from first 10 dimensions)
        (
            cos_th1,
            cos_th2,
            sin_th1,
            sin_th2,
            target_x,
            target_y,
            th1_dot,
            th2_dot,
            _,
            _,
        ) = self._split_observation(obs_physics)

        # Recover angles from sin/cos
        th1 = self._angles_from_sin_cos(sin_th1, cos_th1)
        th2 = self._angles_from_sin_cos(sin_th2, cos_th2)

        # Build generalized coordinates and velocities
        q = torch.cat([th1, th2], dim=-1)
        qdot = torch.cat([th1_dot, th2_dot], dim=-1)

        # Scale actions to torques
        params = self._get_positive_params()
        torques = torch.stack(
            [
                params["torque_scale_1"] * act_flat[:, 0],
                params["torque_scale_2"] * act_flat[:, 1],
            ],
            dim=-1,
        )

        # Integrate physics
        if self.use_rk4:
            q_next, qdot_next = self._integrate_rk4(q, qdot, torques)
        else:
            q_next, qdot_next = self._integrate_euler(q, qdot, torques)

        # Extract next state
        th1_next = q_next[:, 0:1]
        th2_next = q_next[:, 1:2]
        th1_dot_next = qdot_next[:, 0:1]
        th2_dot_next = qdot_next[:, 1:2]

        # Convert to sin/cos representation
        cos_th1_next = torch.cos(th1_next)
        cos_th2_next = torch.cos(th2_next)
        sin_th1_next = torch.sin(th1_next)
        sin_th2_next = torch.sin(th2_next)

        # Compute fingertip position and distance to target
        fingertip_x, fingertip_y = self._compute_fingertip_position(
            th1_next, th2_next, params
        )
        to_target_x_next = fingertip_x - target_x
        to_target_y_next = fingertip_y - target_y

        # Assemble next observation (target position remains constant)
        next_obs_core = torch.cat(
            [
                cos_th1_next,
                cos_th2_next,
                sin_th1_next,
                sin_th2_next,
                target_x,  # Target doesn't change
                target_y,  # Target doesn't change
                th1_dot_next,
                th2_dot_next,
                to_target_x_next,
                to_target_y_next,
            ],
            dim=-1,
        )

        # If there were extra dimensions, pass them through unchanged
        if extra_dims is not None:
            next_obs = torch.cat([next_obs_core, extra_dims], dim=-1)
        else:
            next_obs = next_obs_core

        return next_obs

    def compute_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        target_next_observation: torch.Tensor,
        loss_reduction: str = "mean",
        target_weight: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute prediction loss between model output and target.

        Args:
            observation: Current observation [B, 10]
            action: Applied action [B, 2]
            target_next_observation: True next observation [B, 10]
            loss_reduction: "mean", "sum", or "none"
            target_weight: Weight for target position prediction (indices 4-5)
                          Since target is constant, we may want to down-weight it

        Returns:
            Loss value
        """
        pred = self.forward(observation, action)

        if pred.shape != target_next_observation.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs target {target_next_observation.shape}"
            )

        # Compute squared error
        err = pred - target_next_observation

        # Split into components
        state_err = err[..., :4]  # Angles (sin/cos)
        target_err = err[..., 4:6]  # Target position (should be ~0)
        velocity_err = err[..., 6:8]  # Angular velocities
        to_target_err = err[..., 8:10]  # Distance to target

        # Weighted loss
        loss = (
            state_err.pow(2).mean(dim=-1)
            + target_weight * target_err.pow(2).mean(dim=-1)
            + velocity_err.pow(2).mean(dim=-1)
            + to_target_err.pow(2).mean(dim=-1)
        )

        if loss_reduction == "mean":
            return loss.mean()
        elif loss_reduction == "sum":
            return loss.sum()
        elif loss_reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown loss_reduction: {loss_reduction}")

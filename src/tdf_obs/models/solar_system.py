"""Solar-system GR-safe checks on tau potential fraction."""

from __future__ import annotations


def epsilon_tau(phi_tau: float, phi_b: float) -> float:
    """epsilon_tau = Phi_tau / Phi_b (dimensionless)."""
    if abs(phi_b) < 1e-30:
        raise ValueError("phi_b must be non-zero for epsilon_tau")
    return phi_tau / phi_b


def passes_gr_safe_limit(epsilon: float, max_allowed: float) -> bool:
    """True if |epsilon| <= max_allowed (GR-safe phenomenological bound)."""
    return abs(epsilon) <= max_allowed

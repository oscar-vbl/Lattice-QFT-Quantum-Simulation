
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass
class SymmetryCheckConfig:
    """Configuration for a specific physical symmetry check."""
    name: str                  # Ex: "Charge", "GaussLaw", "Parity", "Chiral"
    target_value: float = 0.0  # Expected physical value for the symmetry check
    tolerance: float = 0.01    # Tolerance for shot-noise or imperfect VQE
    penalty_weight: Optional[float | str] = None  # If exists, it is added to the Hamiltonian to enforce the symmetry. If "auto", it is automatically set to 10x the Hamiltonian's norm.
# QuantumSimulation/core/types/algorithms.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

@dataclass
class OptimizerConfig:
    """Configuration for the classical optimizer in variational algorithms.
    
    Parameters
    ----------
    method : str
        Optimization method to use (e.g., "COBYLA", "L-BFGS-B", "SLSQP").
    maxiter : int
        Maximum number of iterations for the optimizer.
    tol : float
        Tolerance for the optimizer.
    use_gradient : bool
        Whether to use analytical gradients.
    init_strategy : str
        Strategy for initializing parameters.
    init_max : float
        Maximum value for random initialization.
    options : dict[str, Any]
        Additional options for the optimizer.
    """
    method: str = "COBYLA"
    maxiter: int = 2000
    tol: float = 1e-6
    use_gradient: bool = False
    init_strategy: str = "random_small"
    init_max: float = 0.1
    options: dict[str, Any] = field(default_factory=dict)

@dataclass
class VQEResult:
    """Encapsulates all physical and diagnostic data from a VQE execution.
    
    Parameters
    ----------
    state : QuantumCircuit
        The final quantum circuit after VQE optimization.
    energy : float
        The optimized energy value.
    parameters : np.ndarray
        The optimized parameters.
    history : list[float]
        The optimization history.
    duration : float
        The total execution time.
    success : bool
        Whether the optimization was successful.
    iterations : int | None
        The number of iterations performed.
    """
    state: QuantumCircuit
    energy: float
    parameters: np.ndarray
    history: list[float]
    duration: float
    success: bool
    iterations: int | None
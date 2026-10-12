"""
Types for and Ansatz configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class InitialStateConfig:
    """
    Configuration for the initial physical state preparation.
    
    Parameters
    ----------
    vacuum : bool
        Whether to prepare the physical vacuum state. Default is True.
    staggered : bool
        Whether to use staggered fermion configuration (Kogut-Susskind). Default is True.
    """
    vacuum: bool = True
    staggered: bool = True

@dataclass
class AnsatzConfig:
    """
    Configuration for the variational ansatz circuit.
    
    Parameters
    ----------
    ansatz_type : str
        Type of ansatz to build (e.g., "EfficientSU2", "HVA Balanced").
    initial_state : InitialStateConfig
        Configuration object for the base state.
    entanglement : str
        Type of entanglement topology (e.g., "linear", "full"). Default is "linear".
    reps : int
        Number of repetitions (layers) for the parameterized circuit. Default is 3.
    additional_params : dict[str, Any]
        Extra arguments to pass to the specific ansatz function. Default is empty dict.
    """
    ansatz_type: str
    initial_state: InitialStateConfig
    entanglement: str = "linear"
    reps: int = 3
    additional_params: dict[str, Any] = field(default_factory=dict)
"""
Types for hamiltonian configuration and simulation parameters in quantum simulations.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass
class SchwingerParams:
    """Physical parameters specifically for the Schwinger model.

    Parameters
    ----------
    L: int
        Number of lattice sites
    a: float
        Lattice spacing
    m: float
        Mass of the fermions
    e0: float, optional
        Background electric field at the left of the lattice (default 0)    
    periodic: bool, optional
        Whether to apply periodic boundary conditions (default False)
    """
    L: int
    a: float
    m: float
    e0: float = 0.0
    periodic: bool = False

@dataclass
class HamiltonianConfig:
    """
    Typed configuration for generic Hamiltonian generation.
    
    Parameters
    ----------
    model_type : str
        Identifier of the physical model to generate (e.g., "Schwinger").
    gauge : str
        The gauge choice for the formulation (e.g., "Temporal").
    parameters : SchwingerParams
        Typed parameters for the Schwinger model (other models not implemented yet).
    """
    model_type: str
    gauge: str
    parameters: SchwingerParams  # The raw parameters dictionary to be parsed later

# QuantumSimulation/operators/builder.py
from typing import Any
from .fermions import TotalChargeOperator, PairCreationOperator, GaussOperator
from .gauge import ElectricFieldOperator
from QuantumSimulation.hamiltonians import SchwingerHamiltonianTemporalGauge
from QuantumSimulation.core.types import SchwingerParams

# Registry for operator instantiation
OPERATOR_REGISTRY = {
    "Total_Charge_Operator": TotalChargeOperator,
    "Pair_Creation_Operator": PairCreationOperator,
    "Electric_Field_Operator": ElectricFieldOperator,
    "Gauss_Law_Operator": GaussOperator,
    "Schwinger_Hamiltonian_Temporal_Gauge_Operator": SchwingerHamiltonianTemporalGauge,
}

def build_operator(name: str, params: SchwingerParams) -> Any:
    """
    Instantiates a BaseOperator given its name and physical parameters.
    
    Parameters
    ----------
    name : str
        The string identifier of the operator (e.g., "Total_Charge_Operator").
    params : SchwingerParams
        The dataclass containing the physical lattice parameters (L, m, a, etc.).
        
    Returns
    -------
    BaseOperator
        The instantiated operator.
    """
    if name not in OPERATOR_REGISTRY:
        raise ValueError(f"Operator '{name}' is not registered in the builder.")
        
    return OPERATOR_REGISTRY[name](params)

def get_available_operators() -> list[str]:
    """Returns a list of all registered operator names."""
    return list(OPERATOR_REGISTRY.keys())
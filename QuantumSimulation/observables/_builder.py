# QuantumSimulation/observables/builder.py
from typing import Any
from .charge import TotalChargeObservable
from .electric_field import ElectricFieldObservable
from .energy import EnergyObservable
from .gauss_law_violation import GaussLawViolationObservable
from .pair_creation import PairCreationObservable
from .persistence import PersistenceObservable

from .base import BaseObservable

from QuantumSimulation.operators import BaseOperator

# Registry for observable instantiation
OBSERVABLE_REGISTRY: dict[str, BaseObservable] = {
    "Total_Charge": TotalChargeObservable,
    "Energy": EnergyObservable,
    "Electric_Field": ElectricFieldObservable,
    "Gauss_Law_Violation": GaussLawViolationObservable,
    "Pair_Creation": PairCreationObservable,
    "Persistence": PersistenceObservable,
}

# Primitive types before instance
PRIMITIVE_ROUTING: dict[str, str] = {
    "Total_Charge": "estimator",
    "Energy": "estimator",
    "Electric_Field": "estimator",
    "Gauss_Law_Violation": "estimator",
    "Pair_Creation": "estimator",
    "Persistence": "sampler",
}

def build_observable(
    name: str, 
    operator: BaseOperator = None, 
    reference_state: Any = None, 
    precision: float = None, 
    shots: int = None
) -> BaseObservable:
    """
    Instantiates a BaseObservable given its name and required dependencies.
    
    Parameters
    ----------
    name : str
        The string identifier of the observable.
    base_operator : BaseOperator, optional
        The underlying BaseOperator needed to evaluate an estimator observable.
    reference_state : Any, optional
        Reference state to evaluate a sampler observable.
    precision : float, optional
        The precision requirement for the execution engine.
    shots : int, optional
        Number of shots used for sampler calculations.
        
    Returns
    -------
    BaseObservable
        The instantiated observable.
    """
    if name not in OBSERVABLE_REGISTRY:
        raise ValueError(f"Observable '{name}' is not registered in the builder.")
        
    observable_class = OBSERVABLE_REGISTRY[name]
    primitive_type = PRIMITIVE_ROUTING[name]
    
    # We dynamically pass the arguments based on typical observable signatures
    if primitive_type == "estimator":
        if operator is None:
            raise ValueError(f"Estimator observables like '{name}' require a BaseOperator.")
        return observable_class(operator=operator, precision=precision)
        
    elif primitive_type == "sampler":
        if reference_state is None:
            raise ValueError(f"Sampler observables like '{name}' require a reference_state.")
        return observable_class(reference_state=reference_state, shots=shots)


def get_available_observables() -> list[str]:
    """Returns a list of all registered observable names."""
    return list(OBSERVABLE_REGISTRY.keys())
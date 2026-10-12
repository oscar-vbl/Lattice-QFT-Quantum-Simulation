# QuantumSimulation/core/loaders.py
from typing import Any
from .types.diagnostics import SymmetryCheckConfig
from QuantumSimulation.operators._builder import build_operator
from QuantumSimulation.observables._builder import build_observable

OBSERVABLE_OPERATOR_DEPENDENCIES = {
    "Total_Charge": "Total_Charge_Operator",
    "Energy": (
        "Schwinger_Hamiltonian_Temporal_Gauge_Operator"
    ),
    "Electric_Field": "Electric_Field_Operator",
    "Gauss_Law_Violation": "Gauss_Law_Operator",
    "Pair_Creation": "Pair_Creation_Operator",
    "Persistence": None,
}

def load_symmetry_operators(
    symmetry_params: list[tuple[SymmetryCheckConfig, Any]]
) -> list[Any]:
    """
    Instantiates a list of BaseOperator objects required for Hamiltonian symmetry checks.
    
    Parameters
    ----------
    symmetry_params: list[SymmetryCheckConfig]
        List of tuples SymmetryCheckConfig, SchwingerParams (configurations and physical parameters)
        
    Returns
    -------
    list[Any]
        List of instantiated BaseOperator objects.
    """
    loaded_operators = []
    
    for check_config, params in symmetry_params:
        op_instance = build_operator(check_config.name, params)
        loaded_operators.append(op_instance)
        
    return loaded_operators


def load_validation_observables(
    configs_with_params: list[tuple[SymmetryCheckConfig, Any]], 
    reference_state: Any = None,
    precision: float = None,
    shots: int = None
) -> list[Any]:
    """
    Instantiates a list of BaseObservable objects for state validation.
    Dynamically attempts to build the underlying BaseOperator for Estimator observables,
    and falls back gracefully for Sampler observables that only need a reference_state.
    """
    loaded_observables = []
    
    for check_config, params in configs_with_params:
        # 1. Infer the underlying physical operator name
        base_op_name = check_config.name
        
        # 2. Attempt to build the underlying BaseOperator (will fail safely for Samplers)
        operator_name = OBSERVABLE_OPERATOR_DEPENDENCIES[
            base_op_name
        ]

        base_op_instance = (
            build_operator(operator_name, params)
            if operator_name is not None
            else None
        )
        
        # 3. The Builder routes the correct dependencies based on the primitive type
        obs_instance = build_observable(
            name=check_config.name, 
            operator=base_op_instance, 
            reference_state=reference_state,
            precision=precision,
            shots=shots
        )
        loaded_observables.append(obs_instance)
        
    return loaded_observables

def load_evolution_observables(
    observables_list: tuple[str, ...],
    observable_options: dict[str, dict[str, Any]],
    params: Any,
    hamiltonian: Any = None,
    reference_state: Any = None,
    precision: float = None,
    shots: int = None
) -> list[Any]:
    """
    Instantiates a list of BaseObservable objects for temporal evolution.
    Automatically resolves dependencies like the Hamiltonian (for Energy), 
    Reference State (for Persistence), or builds underlying physical operators.
    """
    loaded_observables = []
    
    for obs_name in observables_list:        
        if obs_name == "Energy":
            base_op_instance = hamiltonian

        else:
            operator_name = OBSERVABLE_OPERATOR_DEPENDENCIES[
                obs_name
            ]

            base_op_instance = (
                build_operator(operator_name, params)
                if operator_name is not None
                else None
            )

        obs_instance = build_observable(
            name=obs_name, 
            operator=base_op_instance, 
            reference_state=reference_state,
            precision=precision,
            shots=shots
        )
        loaded_observables.append(obs_instance)
        
    return loaded_observables
from dataclasses import replace
from typing import Any

from QuantumSimulation.core.types import HamiltonianConfig, SchwingerParams
from QuantumSimulation.Utils import getTimer

from .schwinger import SchwingerHamiltonianTemporalGauge
from QuantumSimulation.operators.base import BaseOperator

def build_hamiltonian(
    config: HamiltonianConfig, 
    override_params: dict[str, Any] | None = None
) -> BaseOperator:
    """
    Routes the Hamiltonian construction based on the HamiltonianConfig configuration.
    Applies quench override parameters if provided,
    which allows for dynamic changes in the Hamiltonian parameters during temporal evolution.

    Parameters
    ----------
    config: HamiltonianConfig
        Typed configuration for the Hamiltonian to be constructed.

    override_params: dict[str, Any], optional
        Dictionary of parameters to overwrite the ones in the configuration for the hamiltonian construction.
        This is useful for quenches in the temporal evolution, where the hamiltonian parameters change after t=0.
        The keys of the dictionary must match the ones in config.parameters type.

    Returns
    -------
    SparsePauliOp
            The hamiltonian operator in the form of a sparse Pauli sum.

    Raises
    ------
    NotImplementedError
        If the model_type or gauge specified in the configuration is not implemented.
    """
    if config.model_type == "Schwinger":
        if config.gauge == "Temporal":
            
            # 1. Get the already typed parameters
            current_params: SchwingerParams = config.parameters
            
            # 2. Apply quench safely over the typed object
            if override_params:
                current_params = replace(current_params, **override_params)
                
            # 3. Dispatch to physics
            return SchwingerHamiltonianTemporalGauge(current_params)
            
        else:
            print(f"{getTimer()} WARNING: Gauge {config.gauge} not implemented.")
            raise NotImplementedError(f"Gauge {config.gauge} not implemented.")
    else:
        print(f"{getTimer()} WARNING: Hamiltonian type {config.model_type} not implemented.")
        raise NotImplementedError(f"Hamiltonian type {config.model_type} not implemented.")
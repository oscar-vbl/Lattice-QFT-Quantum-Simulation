from QuantumSimulation.core.types import HardwareConfig, PrimitiveBundle
from QuantumSimulation.Utils import getTimer

from .aer import create_aer_primitives
from .ideal import create_ideal_primitives
from .ibm_runtime import create_ibm_runtime_primitives

def setup_quantum_primitives(hardware_config: HardwareConfig) -> PrimitiveBundle:
    """
    Set up quantum primitives based on the provided hardware configuration.

    Parameters
    ----------
        hardware_config: HardwareConfig
            Configuration for the quantum hardware.

    Returns
    -------
        PrimitiveBundle
            A bundle containing the configured backend, estimator, sampler, and precision.
    """

    if hardware_config.backend_type in ["Aer", "AerSimulator"]:
        print(f"{getTimer()} INFO: Using Primitives V2 backed by AerSimulator backend.")
        return create_aer_primitives(hardware_config)
    elif hardware_config.backend_type in ["Ideal", "StatevectorEstimator"]:
        print(f"{getTimer()} INFO: Using Primitives V2 backed by ideal backend.")
        return create_ideal_primitives(hardware_config)
    elif hardware_config.backend_type == "IBM_Runtime":
        print(f"{getTimer()} INFO: Using Primitives V2 backed by IBM Runtime backend.")
        return create_ibm_runtime_primitives(hardware_config)
    elif hardware_config.backend_type is None or not hardware_config.backend_type:
        print(f"{getTimer()} INFO: No backend provided.")
        return PrimitiveBundle(
            backend_type=hardware_config.backend_type,
            backend=None,
            estimator=None,
            sampler=None,
            precision=hardware_config.precision
        )
    else:
        raise NotImplementedError(
            f"Backend type {hardware_config.backend_type} not recognized."
        )
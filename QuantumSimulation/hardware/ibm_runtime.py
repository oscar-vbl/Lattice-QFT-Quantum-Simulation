# QuantumSimulation/hardware/ibm_runtime.py
import os
from QuantumSimulation.core.types import HardwareConfig, PrimitiveBundle
from QuantumSimulation.Utils import getTimer

def create_ibm_runtime_primitives(config: HardwareConfig) -> PrimitiveBundle:
    """
    Connects to Qiskit Runtime Service and configures hardware primitives.
    Authenticates automatically via the QISKIT_IBM_TOKEN environment variable if present.

    Parameters
    ----------
        config: HardwareConfig
            Configuration for the quantum hardware.
    
    Returns
    -------
        PrimitiveBundle
            A bundle containing the configured AerSimulator backend, EstimatorV2, SamplerV2, and precision.
    """
    # lazy imports to not force the dependency on ibm_runtime
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2
    from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2

    print(f"{getTimer()} INFO: Connecting to IBM Qiskit Runtime Service...")
    
    # Search token in environment variables
    ibm_token = os.getenv("QISKIT_IBM_TOKEN")
    
    if ibm_token:
        # If the token exists in the environment, log in explicitly (Ideal for Docker/CI)
        service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)
    else:
        # If not, rely on the user having saved their credentials to disk (Qiskit default)
        service = QiskitRuntimeService()

    if config.hardware_name == "least_busy":
        backend = service.least_busy(operational=True, simulator=False)
    else:
        backend = service.backend(config.hardware_name)

    print(f"{getTimer()} INFO: Target hardware selected: {backend.name}")

    estimator = RuntimeEstimatorV2(mode=backend)
    sampler = RuntimeSamplerV2(mode=backend)

    # Apply advanced options such as mitigation level (ZNE)
    resilience_level = config.options.get("resilience_level", 1)
    estimator.options.resilience_level = resilience_level

    return PrimitiveBundle(
        backend_type=config.backend_type,
        backend=backend,
        estimator=estimator,
        sampler=sampler,
        precision=config.precision,
        shots=config.shots
    )
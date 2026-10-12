# QuantumSimulation/hardware/ideal.py
import numpy as np
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

from QuantumSimulation.core.types import HardwareConfig, PrimitiveBundle

def create_ideal_primitives(config: HardwareConfig) -> PrimitiveBundle:
    """
    Create a PrimitiveBundle for Qiskit Ideal based on the provided HardwareConfig.

    Parameters
    ----------
        config: HardwareConfig
            Configuration for the quantum hardware.
    
    Returns
    -------
        PrimitiveBundle
            A bundle containing the configured Ideal backend, StatevectorEstimator, StatevectorSampler, and precision.
    """
    
    if config.shots is not None:
        estimator_options = {"default_precision": 1 / np.sqrt(config.shots)}
        sampler_options   = {"default_shots": config.shots}
    else:
        estimator_options = {}
        sampler_options   = {}

    estimator = StatevectorEstimator(**estimator_options)
    sampler   = StatevectorSampler(**sampler_options)
    backend   = None

    return PrimitiveBundle(
        backend_type=None,
        backend=backend,
        estimator=estimator,
        sampler=sampler,
        precision=config.precision,
        shots=config.shots
    )

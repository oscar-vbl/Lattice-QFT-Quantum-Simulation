# QuantumSimulation/hardware/aer.py
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_aer.noise import NoiseModel

from QuantumSimulation.core.types import HardwareConfig, PrimitiveBundle
from typing import Any

from .noise import build_noise_model

def create_aer_primitives(config: HardwareConfig) -> PrimitiveBundle:
    """
    Create a PrimitiveBundle for Qiskit Aer based on the provided HardwareConfig.

    Parameters
    ----------
        config: HardwareConfig
            Configuration for the quantum hardware.
    
    Returns
    -------
        PrimitiveBundle
            A bundle containing the configured AerSimulator backend, EstimatorV2, SamplerV2, and precision.
    """
    
    aer_noise_model = build_noise_model(config.noise_model_config)
    aer_backend = AerSimulator(noise_model=aer_noise_model)
    
    estimator_options = create_estimator_options_aer(config, aer_noise_model)
    estimator = EstimatorV2(options=estimator_options)
    
    shots, sampler_options = create_sampler_options_aer(config, aer_noise_model)
    sampler = SamplerV2(default_shots=shots, options=sampler_options)

    return PrimitiveBundle(
        backend_type="AerSimulator",
        backend=aer_backend,
        estimator=estimator,
        sampler=sampler,
        precision=config.precision,
        shots=shots 
    )

def create_estimator_options_aer(
        config: HardwareConfig,
        aer_noise_model: NoiseModel | None = None
) -> dict[str, Any]:
    """
    Create options dictionary for the EstimatorV2 based on the provided HardwareConfig and optional Aer noise model.
    
    Parameters
    ----------
        config: HardwareConfig
            Configuration for the quantum hardware.
        aer_noise_model: NoiseModel or None
            Optional Qiskit Aer noise model to be applied to the EstimatorV2.
    
    Returns
    -------
        dict[str, Any]
            A dictionary containing the configured options for the EstimatorV2.
    """
    estimator_options: dict[str, Any] = {}
    if aer_noise_model is not None:
        estimator_options["backend_options"] = {"noise_model": aer_noise_model}
    else:
        estimator_options["backend_options"] = {}

    if config.options:
        estimator_options["backend_options"].update(config.options)
        
    if config.precision is not None:
        estimator_options["default_precision"] = config.precision
    elif config.shots is not None:
        estimator_options["default_precision"] = 1.0 / np.sqrt(config.shots)
    
    return estimator_options


def create_sampler_options_aer(
        config: HardwareConfig,
        aer_noise_model: NoiseModel | None = None
) -> tuple[int | None, dict[str, Any]]:
    """
    Create options dictionary for the SamplerV2 based on the provided HardwareConfig and optional Aer noise model.
    
    Parameters
    ----------
        config: HardwareConfig
            Configuration for the quantum hardware.
        aer_noise_model: NoiseModel or None
            Optional Qiskit Aer noise model to be applied to the SamplerV2.
    
    Returns
    -------
        tuple[int | None, dict[str, Any]]
            A tuple containing the default shots and the configured options for the SamplerV2.
    """
    sampler_options: dict[str, Any] = {}
    if aer_noise_model is not None:
        sampler_options["backend_options"] = {"noise_model": aer_noise_model}
    else:
        sampler_options["backend_options"] = {}
        
    if config.options:
        sampler_options["backend_options"].update(config.options)
        
    if config.shots is not None:
        shots = config.shots
    else:
        shots = 1024  # Default shots if not specified

    return shots, sampler_options
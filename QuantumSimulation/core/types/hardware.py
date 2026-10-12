"""
Types for hardware configuration and noise modeling in quantum simulations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class NoiseErrorConfig:
    """
    Configuration for a specific error in the noise model.
    
    Parameters
    ----------
    error_type : str
        Type of the quantum error (e.g., "Depolarizing", "ThermalRelaxation").
    num_qubits : int
        Number of qubits the error applies to. Default is 1.
    probability : float
        Probability of the error occurring. Default is 0.01.
    gates : list[str]
        List of gate names the error applies to. Default is an empty list.
    """
    error_type: str                  # Ej: "Depolarizing", "ThermalRelaxation"
    num_qubits: int = 1
    probability: float = 0.01
    gates: list[str] = field(default_factory=list)

@dataclass
class NoiseConfig:
    """
    Global configuration for the noise model.
    
    Parameters
    ----------
    errors : list[NoiseErrorConfig]
        List containing the configurations for all specific errors in the model.
        Default is an empty list.
    """
    errors: list[NoiseErrorConfig] = field(default_factory=list)

@dataclass
class HardwareConfig:
    """
    Typed configuration for quantum hardware.
    
    Parameters
    ----------
    backend_type : Optional[str]
        Type of backend to use (e.g., "StatevectorEstimator", "AerSimulator", "IBM_Runtime").
        Default is None (direct matrix operations).
    shots : Optional[int]
        Number of shots for execution. Default is None.
    precision : Optional[float]
        Target precision for the estimator. Default is None.
    hardware_name : str
        Name of the target physical or simulated hardware (only used in backend_type "IBM_Runtime"). Default is "least_busy".
    noise_model_config : Optional[NoiseConfig]
        Configuration object defining the noise model. Default is None.
    options : dict[str, Any]
        Additional backend-specific execution options. Default is an empty dictionary.
    """
    backend_type: Optional[str] = None
    shots: Optional[int] = None
    precision: Optional[float] = None
    hardware_name: str = "least_busy"
    noise_model_config: Optional[NoiseConfig] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrimitiveBundle:
    """
    Main data class encapsulating hardware execution and Primitives V2 configuration.
    
    Parameters
    ----------
    backend_type : Optional[str]
        String identifier of the active backend.
    backend : Optional[Any]
        The instantiated backend object (e.g., AerSimulator or IBM_Runtime).
    estimator : Optional[Any]
        The instantiated Qiskit EstimatorV2 primitive.
    sampler : Optional[Any]
        The instantiated Qiskit SamplerV2 primitive.
    precision : Optional[float]
        The final target precision configured for the primitives.
    shots : Optional[int]
        The final target shots configured for the sampler primitives.
    """
    backend_type: Optional[str]
    backend: Optional[Any]
    estimator: Optional[Any]
    sampler: Optional[Any]
    precision: Optional[float]
    shots: int | None = None
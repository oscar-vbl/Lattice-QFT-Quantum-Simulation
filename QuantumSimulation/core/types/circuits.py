# QuantumSimulation/core/types/circuits.py
"""
Types for quantum circuit building and gate configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class QubitRef:
    """
    Reference to a specific qubit inside a quantum circuit.
    
    Parameters
    ----------
    index : int
        The integer index of the qubit.
    is_ancilla : bool
        Whether this index refers to the ancilla register. Default is False.
    """
    index: int
    is_ancilla: bool = False


@dataclass
class GateConfig:
    """
    Configuration for a single quantum gate.
    
    Parameters
    ----------
    gate_type : str
        String identifier for the quantum gate (e.g., "X", "H", "CNOT", "U", "MEASURE").
    qubit : Optional[QubitRef]
        Reference to the target qubit for single-qubit gates or measurements.
    control : Optional[QubitRef]
        Reference to the control qubit for two-qubit gates (like CNOT).
    target : Optional[QubitRef]
        Reference to the target qubit for two-qubit gates (like CNOT).
    params : list[float]
        Universal list of parameters for parametric gates. 
        Provides 0 parameters for fixed gates, 1 parameter for rotations (RX, RY, RZ), 
        or 3 parameters for U gates.
    classical_bit : Optional[int]
        Index of the classical bit where the measurement result will be stored.
    """
    gate_type: str
    qubit: Optional[QubitRef] = None
    control: Optional[QubitRef] = None
    target: Optional[QubitRef] = None
    params: list[float] = field(default_factory=list)
    classical_bit: Optional[int] = None


@dataclass
class CircuitConfig:
    """
    Complete configuration for building a QuantumCircuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    num_ancillas : int
        Number of ancilla qubits. Default is 0.
    num_measurements : Optional[int]
        Number of classical bits for measurement. Default is None.
    gates : list[GateConfig]
        List of gate configurations to be applied in the circuit.
    """
    num_qubits: int
    num_ancillas: int = 0
    num_measurements: int = 0
    gates: list[GateConfig] = field(default_factory=list)
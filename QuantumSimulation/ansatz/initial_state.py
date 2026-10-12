from qiskit import QuantumCircuit

from QuantumSimulation.circuits.builder import build_circuit

from QuantumSimulation.Utils import getTimer

from QuantumSimulation.core.types import InitialStateConfig
from QuantumSimulation.core.types import CircuitConfig, GateConfig, QubitRef


def build_initial_state(num_qubits: int, config: InitialStateConfig) -> QuantumCircuit:
    """
    Builds the non-parameterized initial state circuit (e.g., staggered vacuum).

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the circuit.
    config: InitialStateConfig
        Configuration for the initial state preparation.
    
    Returns
    -------
    QuantumCircuit
        A Qiskit QuantumCircuit representing the initial state.
    
    Raises
    ------
    NotImplementedError
        If the initial state configuration is not supported (e.g., non-vacuum or non-staggered).
    """
    if not config.vacuum:
        print(f"{getTimer()} WARNING: Initial state not set to vacuum is not implemented.")
        raise NotImplementedError("Ansatz state for not vacuum state not implemented.")
        
    if not config.staggered:
        print(f"{getTimer()} WARNING: Ansatz state for not staggered fermions not implemented.")
        raise NotImplementedError("Ansatz state for not staggered fermions not implemented.")

    # Apply X gates to even sites for staggered fermions
    gate_list = [
        GateConfig(gate_type="X", qubit=QubitRef(index=i)) 
        for i in range(0, num_qubits, 2)
    ]
    
    circuit_config = CircuitConfig(num_qubits=num_qubits, gates=gate_list)
    return build_circuit(circuit_config)
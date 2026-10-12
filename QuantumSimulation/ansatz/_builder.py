# QuantumSimulation/vqe/builder.py
from qiskit import QuantumCircuit

from QuantumSimulation.Utils import getTimer

from QuantumSimulation.core.types import AnsatzConfig, PrimitiveBundle
from QuantumSimulation.operators import BaseOperator

from .initial_state import build_initial_state
from qiskit.circuit.library import efficient_su2, excitation_preserving
from .hva import build_schwinger_hva_balanced, build_schwinger_hva_full, build_schwinger_hva

def build_parametrized_ansatz(
    num_qubits: int, 
    config: AnsatzConfig, 
    hamiltonian: BaseOperator | None = None,
    primitives: PrimitiveBundle | None = None
) -> QuantumCircuit:
    """
    Constructs the full ansatz by composing the initial state and the parameterized layers.

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the system.
    config: AnsatzConfig
        Configuration for the ansatz, including type, repetitions, and additional parameters.
    hamiltonian: BaseOperator | None
        The Hamiltonian of the system, required for HVA types.
    
    Returns
    -------
    QuantumCircuit
        The full ansatz circuit, including the initial state and parameterized layers.
    
    Raises
    ------
    NotImplementedError
        If the ansatz type specified in the configuration is not implemented.
    ValueError
        If the Hamiltonian is not provided for HVA ansatz types.
    TypeError
        If the ansatz function does not return a QuantumCircuit.
    """
    
    # 1. Build initial state (Vacuum)
    initial_circuit = build_initial_state(num_qubits, config.initial_state)

    # 2. Select parameterized function
    ansatz_type = config.ansatz_type
    
    # Qiskit heuristics
    if ansatz_type == "EfficientSU2":
        ansatz_func = efficient_su2
    elif ansatz_type == "ExcitationPreserving":
        ansatz_func = excitation_preserving
    #elif ansatz_type == "TwoLocal":
    #    ansatz_func = n_local

    # Custom HVAs
    elif ansatz_type == "HVA":
        ansatz_func = build_schwinger_hva_balanced
    elif ansatz_type == "HVA Full":
        ansatz_func = build_schwinger_hva_full
    elif ansatz_type == "HVA Simple":
        ansatz_func = build_schwinger_hva
    else:
        print(f"{getTimer()} WARNING: Ansatz type {ansatz_type} not implemented.")
        raise NotImplementedError(f"Ansatz type {ansatz_type} not implemented.")

    # 3. Prepare parameters
    ansatz_params = {
        "num_qubits": num_qubits,
        "entanglement": config.entanglement,
        "reps": config.reps,
        **config.additional_params
    }
    
    # Inject hamiltonian explicitly for HVA types
    if "HVA" in ansatz_type:
        if hamiltonian is not None:
            ansatz_params["hamiltonian"] = hamiltonian
        else:
            print(f"{getTimer()} WARNING: Hamiltonian must be provided for HVA ansatz.")
            raise ValueError("Hamiltonian must be provided for HVA ansatz.")

    print(f"{getTimer()} INFO: Using ansatz {ansatz_type} with reps: {config.reps}")

    # 4. Build and compose
    parameterized_circuit = ansatz_func(**ansatz_params)
    
    if not isinstance(parameterized_circuit, QuantumCircuit):
        raise TypeError("Ansatz function did not return a QuantumCircuit.")

    # Evolve from initial vacuum
    full_ansatz_circuit = initial_circuit.compose(parameterized_circuit)
    return full_ansatz_circuit
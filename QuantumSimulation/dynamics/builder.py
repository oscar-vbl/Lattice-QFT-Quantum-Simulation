# QuantumSimulation/dynamics/builder.py

from qiskit import QuantumCircuit
from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter

from QuantumSimulation.core.types import EvolutionConfig, PrimitiveBundle
from QuantumSimulation.operators.base import BaseOperator

from QuantumSimulation.Utils import getTimer

from QuantumSimulation.circuits.transpilation import apply_transpilation

def build_evolution_circuits(
    hamiltonian: BaseOperator, 
    config: EvolutionConfig, 
    primitives: PrimitiveBundle, 
    num_qubits: int
) -> dict[str, QuantumCircuit]:
    """
    Builds and optionally transpiles the evolution step circuits for each trajectory variant.
    """
    evolution_circuits = {}
    
    # Select synthesis method
    if config.synthesis == "suzuki_trotter":
        synthesis = SuzukiTrotter(**config.synthesis_options)
    else:
        print(f"{getTimer()} WARNING: Synthesis method {config.synthesis} not implemented.")
        raise NotImplementedError(f"Synthesis method {config.synthesis} not implemented.")

    print(f"{getTimer()} INFO: Building evolution circuits using {config.synthesis} synthesis.")

    for variant in config.variants:
        current_time_step = config.base_step * variant.step_multiplier
        
        evolution_gate = PauliEvolutionGate(
            hamiltonian.to_qiskit(sparse=True), 
            time=current_time_step, 
            synthesis=synthesis
        )
        
        step_circuit = QuantumCircuit(num_qubits)
        step_circuit.append(evolution_gate, range(num_qubits))
        
        # Hardware Transpilation or Ideal Decomposition
        if primitives.backend is not None and primitives.backend_type not in ["StatevectorEstimator", None]:
            print(f"{getTimer()} INFO: Transpiling Trotter variant '{variant.key}' to ISA...")
            # We assume initial_layout logic will be injected or handled globally later
            transpiled_qc, _ = apply_transpilation(primitives.backend, step_circuit)
            evolution_circuits[variant.key] = transpiled_qc
        else:
            evolution_circuits[variant.key] = step_circuit.decompose().decompose()
            
    return evolution_circuits
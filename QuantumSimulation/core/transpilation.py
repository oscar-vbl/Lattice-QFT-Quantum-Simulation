from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.transpiler.layout import TranspileLayout

def apply_transpilation(
            backend: AerSimulator | QiskitRuntimeService,
            circuit: QuantumCircuit,
            initial_layout: TranspileLayout | None = None,
            layout_circuits: list[QuantumCircuit]=[]):
    """
    Transpile a quantum circuit to the target backend's instruction set architecture (ISA) using Qiskit's PassManager.
    
    Parameters
    ----------
    backend : AerSimulator or QiskitRuntimeService
        The target backend to which the circuit will be transpiled.
    circuit : QuantumCircuit
        The quantum circuit to be transpiled.
    initial_layout : TranspileLayout or None, optional
        Hardware layout and routing information added by the transpiler.
    layout_circuits : list of QuantumCircuit, optional
        Additional circuits that depend on the qubit layout (e.g., Hamiltonian circuits) that should be adapted to the transpiled circuit's layout. Default is an empty list.

    Returns
    -------
    transpiled_circuit : QuantumCircuit
        The transpiled quantum circuit compatible with the target backend.
    tp_layout_circuits : list of QuantumCircuit
        The layout-dependent circuits adapted to the transpiled circuit's qubit layout. If no layout was applied, returns the original layout_circuits.
    """
    # Generate PassManager with the specified backend (AerSimulator or IBM Runtime)
    pm = generate_preset_pass_manager(
            optimization_level=1,
            backend=backend,
            initial_layout=initial_layout
        )

    # Transpile ansatz to the target backend's ISA using the PassManager
    transpiled_circuit = pm.run(circuit)

    # When transpiling, manager can reorder physical qubits
    # Some other circuits' topology (ex. Hamiltonian) must be adapted
    if transpiled_circuit.layout is not None and layout_circuits is not None:
        tp_layout_circuits = []
        for layout_circuit in layout_circuits:
            layout_circuit = layout_circuit.apply_layout(
                    transpiled_circuit.layout
                )
            tp_layout_circuits.append(layout_circuit)
        return transpiled_circuit, tp_layout_circuits
    else:
        return transpiled_circuit, layout_circuits
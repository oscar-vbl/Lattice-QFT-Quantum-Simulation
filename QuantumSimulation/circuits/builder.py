from qiskit import QuantumCircuit, AncillaRegister, ClassicalRegister, QuantumRegister
from qiskit.circuit.exceptions import CircuitError

from QuantumSimulation.core.types import CircuitConfig, GateConfig, QubitRef

def build_circuit(config: CircuitConfig) -> QuantumCircuit:
    """
    Build a quantum circuit given a CircuitConfig configuration object.

    Parameters
    ----------
    config: CircuitConfig
        Configuration object specifying the circuit to be built. It should include the following
        - num_qubits: int, number of qubits in the circuit
        - num_ancillas: int, number of ancilla qubits (optional)
        - num_measurements: int, number of classical bits for measurement (optional)
        - gates: list of GateConfig objects, each specifying a gate and its parameters

    Returns
    -------
    QuantumCircuit
        The quantum circuit built according to the configuration.
    """
    # 1. Initialize Registers
    q_reg = QuantumRegister(config.num_qubits, "q")
    
    if config.num_measurements:
        c_reg = ClassicalRegister(config.num_measurements, "c")
        circuit = QuantumCircuit(q_reg, c_reg)
    else:
        circuit = QuantumCircuit(q_reg)
        
    ancilla_reg = None
    if config.num_ancillas > 0:
        ancilla_reg = AncillaRegister(config.num_ancillas, "ancilla")
        circuit.add_register(ancilla_reg)

    # 2. Add Gates
    for gate in config.gates:
        _add_gate(circuit, gate, ancilla_reg)

    return circuit

def _resolve_qubit(qubit_ref: QubitRef, ancilla_reg: AncillaRegister | None):
    """Helper to safely resolve whether a target is a standard qubit or an ancilla."""
    if qubit_ref.is_ancilla:
        if ancilla_reg is None:
            raise ValueError("Ancilla qubit requested but no ancilla register exists.")
        return ancilla_reg[qubit_ref.index]
    else:
        return qubit_ref.index


def _add_gate(circuit: QuantumCircuit, gate: GateConfig, ancilla_reg: AncillaRegister | None = None):
    """
    Router function to delegate gate addition based on gate classification.
    """
    gate_type = gate.gate_type.upper()

    if gate_type == "MEASURE":
        _add_measurement(circuit, gate, ancilla_reg)
    elif gate.control is not None and gate.target is not None:
        _add_two_qubit_gate(circuit, gate, ancilla_reg)
    elif gate.qubit is not None:
        _add_single_qubit_gate(circuit, gate, ancilla_reg)
    else:
        raise ValueError(f"Invalid gate configuration for {gate_type}: missing target qubits.")

def _add_single_qubit_gate(circuit: QuantumCircuit, gate: GateConfig, ancilla_reg: AncillaRegister | None):
    """
    Handles all 1-qubit gates. 
    Uses *gate.params to automatically unpack 0, 1, or 3 angles.
    """
    target = _resolve_qubit(gate.qubit, ancilla_reg)
    gate_type = gate.gate_type.upper()

    if gate_type == "H":
        circuit.h(target)
    elif gate_type == "X":
        circuit.x(target)
    elif gate_type == "SDG":
        circuit.sdg(target)
    elif gate_type in ["RX", "RY", "RZ"]:
        # Ensure we have exactly 1 parameter for rotation gates
        if len(gate.params) != 1:
            raise ValueError(f"{gate_type} gate requires exactly 1 parameter.")
            
        if gate_type == "RX":
            circuit.rx(gate.params[0], target)
        elif gate_type == "RY":
            circuit.ry(gate.params[0], target)
        elif gate_type == "RZ":
            circuit.rz(gate.params[0], target)
            
    elif gate_type == "U":
        # Ensure we have exactly 3 parameters for U gate
        if len(gate.params) != 3:
            raise ValueError("U gate requires exactly 3 parameters.")
        # alpha, beta, gamma
        circuit.u(*gate.params, target)
    else:
        print(f"ERROR: Unsupported single-qubit gate type: {gate_type}")

def _add_two_qubit_gate(circuit: QuantumCircuit, gate: GateConfig, ancilla_reg: AncillaRegister | None):
    """
    Handles all 2-qubit controlled gates.
    """
    control = _resolve_qubit(gate.control, ancilla_reg)
    target = _resolve_qubit(gate.target, ancilla_reg)
    gate_type = gate.gate_type.upper()

    if gate_type == "CNOT" or gate_type == "CX":
        circuit.cx(control, target)
    # Future additions: CRZ, CZ, SWAP, etc.
    else:
        print(f"ERROR: Unsupported two-qubit gate type: {gate_type}")

def _add_measurement(circuit: QuantumCircuit, gate: GateConfig, ancilla_reg: AncillaRegister | None):
    """
    Handles measurement operations with fallback logic.
    """
    target = _resolve_qubit(gate.qubit, ancilla_reg)
    
    try:
        circuit.measure(target, gate.classical_bit)
    except CircuitError as e:
        if len(circuit.clbits) > 0:
            circuit.measure(target, circuit.clbits[gate.classical_bit])
        else:
            print(f"ERROR adding measurement: {e}. Adding fallback classical register.")
            circuit.add_register(ClassicalRegister(len(circuit.qubits), "c_fallback"))
            circuit.measure(target, circuit.clbits[gate.classical_bit])
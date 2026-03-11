import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, AncillaRegister, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RXGate, RYGate, RZGate, CXGate, UGate
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info import Statevector

def buildCircuit(configuration):
    '''
    Build quantum circuit given a configuration dictionary.

    Params:
        *configuration*: dict

    Entries of configuration:
        - "QubitsNumber": int, number of qubits in the circuit
        - "AncillaQubits": int, number of ancilla qubits (optional)
        - "Gates": list of dicts, each dict specifies a gate and its parameters

    Params of each gate dict:
        - "gate": str, type of gate ("CNOT", "RZ", etc.)
        - Entries for single-qubit gates:
            - "qubit": qubit-dict
        - Entries for double qubit gates:
            - "control": qubit-dict for control qubit
            - "target": qubit-dict for target qubit

    Params of each qubit-dict:
        - "Number": number of the qubit
        - "Ancilla": if it's an ancillary qubit (optional)
        - Specific parameters depending on the gate type
    '''
    numQubits     = configuration["QubitsNumber"]
    ancillaQubits = configuration.get("AncillaQubits", 0)
    measureQubits = configuration.get("MeasurementQubits", None)

    if measureQubits: circuit = QuantumCircuit(QuantumRegister(numQubits), ClassicalRegister(measureQubits))
    else:             circuit = QuantumCircuit(QuantumRegister(numQubits))
    if ancillaQubits > 0:
        ancilla = AncillaRegister(ancillaQubits, 'ancilla')
        circuit.add_register(ancilla)
    else:
        ancilla = None
    
    for gate in configuration["Gates"]:
        addGate(circuit, gate, ancilla)

    return circuit

def addGate(circuit: QuantumCircuit, gate, ancilla=None):
    gateType = gate["gate"]
    if not ancilla: ancilla = circuit.ancillas
    if gateType == "CNOT":
        # Control qubit of CNOT gate
        controlQubitNum = gate["control"]["Number"]
        isAncilla       = gate["control"].get("Ancilla", False)
        if isAncilla: controlQubit = ancilla[controlQubitNum]
        else:         controlQubit = controlQubitNum
        # Target qubit of CNOT gate
        targetQubitNum = gate["target"]["Number"]
        isAncilla      = gate["target"].get("Ancilla", False)
        if isAncilla: targetQubit = ancilla[targetQubitNum]
        else:         targetQubit = targetQubitNum
        # Add CNOT gate
        circuit.cx(controlQubit, targetQubit)

    elif gateType == "H":
        # Hadamard gate
        qubitNum  = gate["qubit"]["Number"]
        isAncilla = gate["qubit"].get("Ancilla", False)
        if isAncilla: qubit = ancilla[qubitNum]
        else:         qubit = qubitNum
        circuit.h(qubit)

    elif gateType == "X":
        # X gate (flips the state of the qubit)
        qubitNum  = gate["qubit"]["Number"]
        isAncilla = gate["qubit"].get("Ancilla", False)
        if isAncilla: qubit = ancilla[qubitNum]
        else:         qubit = qubitNum
        circuit.x(qubit)

    elif gateType == "SDG":
        # S-Dagger gate
        qubitNum  = gate["qubit"]["Number"]
        isAncilla = gate["qubit"].get("Ancilla", False)
        if isAncilla: qubit = ancilla[qubitNum]
        else:         qubit = qubitNum
        circuit.sdg(qubit)

    elif gateType == "RX":
        # RX (phase shift around x-axis) gate
        qubitNum  = gate["qubit"]["Number"]
        isAncilla = gate["qubit"].get("Ancilla", False)
        if isAncilla: qubit = ancilla[qubitNum]
        else:         qubit = qubitNum
        circuit.rx(gate["angle"], qubit)

    elif gateType == "RY":
        # RY (phase shift around y-axis) gate
        qubitNum  = gate["qubit"]["Number"]
        isAncilla = gate["qubit"].get("Ancilla", False)
        if isAncilla: qubit = ancilla[qubitNum]
        else:         qubit = qubitNum
        circuit.ry(gate["angle"], qubit)

    elif gateType == "RZ":
        # RZ (phase shift around z-axis) gate
        qubitNum  = gate["qubit"]["Number"]
        isAncilla = gate["qubit"].get("Ancilla", False)
        if isAncilla: qubit = ancilla[qubitNum]
        else:         qubit = qubitNum
        circuit.rz(gate["angle"], qubit)

    elif gateType == "U":
        # U gate, defined by 3 angles: alpha, beta, gamma
        qubitNum  = gate["qubit"]["Number"]
        isAncilla = gate["qubit"].get("Ancilla", False)
        if isAncilla: qubit = ancilla[qubitNum]
        else:         qubit = qubitNum
        circuit.u(*gate["angles"], qubit)

    elif gateType == "Measure":
        # Measurement gate
        qubitNum  = gate["qubit"]["Number"]
        isAncilla = gate["qubit"].get("Ancilla", False)
        if isAncilla: qubit = ancilla[qubitNum]
        else:         qubit = qubitNum
        classicalBit = gate["classicalBit"]
        try:
            circuit.measure(qubit, classicalBit)
        except CircuitError as e:
            if len(circuit.clbits) > 0:
                circuit.measure(qubit, circuit.clbits[classicalBit])
            else:
                print(f"ERROR adding measurement: {e}, adding classical register.")
                circuit.add_register(ClassicalRegister(len(circuit.qubits)))
    # Add other gates as needed
    else:
        raise print(f"ERROR: Unsupported gate type: {gateType}")
    

def get_statevector_from_counts(counts):
    '''
    Convert measurement counts to a Statevector.

    WARNING: Only for low number of qubits, as it reconstructs the full statevector.

    Reconstructs a state from the output measurements of an experiment

    Params:
        *counts*: dict, measurement counts from a quantum job

    Returns:
        qiskit.quantum_info.Statevector
    '''
    # 1. Convert counts to probabilities
    shots = sum(counts.values())
    probs = {label: counts[label]/shots for label in counts}

    # 2. Determine number of qubits
    n = len(next(iter(counts)))  # length of bitstrings
    dim = 2**n

    # 3. Initialize amplitude vector
    amps = np.zeros(dim, dtype=complex)

    # 4. Fill amplitudes = sqrt(prob)
    for bitstring, p in probs.items():
        index = int(bitstring, 2)  # bitstring → basis index
        amps[index] = np.sqrt(p)   # real, non-negative amplitude

    # 5. Build statevector
    sv = Statevector(amps)
    return sv

if __name__ == "__main__":
    numQubits = 3
    numAncilla = 1
    dt = 0.1
    gates = []
    for qubit in range(numQubits):
        gates += [
            {'gate': 'CNOT', 'control': {"Number": qubit}, 'target': {"Number": 0, "Ancilla":True}},
        ]
    gates += [{'gate': 'RZ', 'qubit': {"Number": 0, "Ancilla":True}, 'angle': 2*dt}]
    for qubit in reversed(range(numQubits)):
        gates += [
            {'gate': 'CNOT', 'control': {"Number": qubit}, 'target': {"Number": 0, "Ancilla":True}},
        ]
    configuration = {
        "QubitsNumber": numQubits,
        "AncillaQubits": numAncilla,
        "Gates": gates
    }
    circuit = buildCircuit(configuration)
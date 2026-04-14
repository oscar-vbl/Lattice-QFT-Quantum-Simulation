import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, AncillaRegister, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RXGate, RYGate, RZGate, CXGate, UGate
from qiskit.circuit.exceptions import CircuitError


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

def addGate(
        circuit: QuantumCircuit,
        gate: dict,
        ancilla=None):
    '''
    Add a gate to the circuit according to the gate configuration.
    Params:
    - circuit: QuantumCircuit, the circuit to which the gate will be added.
    - gate: dict, configuration of the gate to be added. Must include "gate" key specifying the type of gate and other keys depending on the gate type.
    - ancilla: AncillaRegister, optional, the ancilla register if needed for the gate.
    '''
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
    
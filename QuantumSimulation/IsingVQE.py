# Basic VQE solver for the Transverse Ising model using Qiskit

from circuitBuilder import buildCircuit, addGate
from Utils import sortEigenstates, drawCircuitLatex
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import GenericBackendV2

import numpy as np

def sigma_expectation(counts, qubit_indexes):
    shots = sum(counts.values())
    expected = 0
    
    for qubit_label, count in counts.items():
        # Doing these because Qiskit uses little-endian bitstrings
        # Returns reversed, so, to select from a string like '001'
        # the first qubit (index 0, value '1'), we need to do this
        rev_qubit_label = qubit_label[::-1]  # reverse bitstring
        ind_label_expectation = count / shots
        for qubit_index in qubit_indexes:
            qubit_state = rev_qubit_label[qubit_index]   # reverse because Qiskit uses little-endian
            if qubit_state == '0': value = +1
            else:                  value = -1
            ind_label_expectation *= value
        expected += ind_label_expectation
    
    return expected

def getVQEAnsatzConfig(numQubits, theta_angles, numAncilla=0):
    '''
    Builds a simple VQE ansatz circuit configuration for the transverse Ising model.
    First two theta angles are for an initial pseudo-Hadamard on the first qubit,
    the other angles (one per qubit) is for RY rotations on each qubit.'''
    configuration = {
        "QubitsNumber":      numQubits,
        "AncillaQubits":     numAncilla,
        "MeasurementQubits": numQubits,
        "Gates": []
    }

    # First: Pseudo-Hadamard gate on first qubit
    configuration["Gates"] += [{'gate': 'RZ', 'qubit': {"Number": 0}, 'angle': theta_angles[0]}]
    configuration["Gates"] += [{'gate': 'RY', 'qubit': {"Number": 0}, 'angle': theta_angles[1]}]

    # Second: Entangling CNOTs
    for qubit in range(1, numQubits):
        configuration["Gates"] += [
            {'gate': 'CNOT', 'control': {"Number": 0}, 'target': {"Number": qubit}},
        ]
    
    # Third: RY rotations on each qubit
    used_theta = 2
    for qubit in range(numQubits):
        configuration["Gates"] += [
            {'gate': 'RY', 'qubit': {"Number": qubit}, 'angle': theta_angles[used_theta+qubit]},
        ]
    
    return configuration

class VQESolver():
    def __init__(self, numQubits, hamiltonian, ansatz_circuit=None):
        self.numQubits = numQubits
        self.hamiltonian = hamiltonian
        self.backend = GenericBackendV2(num_qubits=self.numQubits)
        self.ansatz_circuit = ansatz_circuit



    def getEnergy(self, theta_angles, shotsNum=1024):
        if not self.ansatz_circuit:
            configuration = getVQEAnsatzConfig(self.numQubits, theta_angles)
            ansatz_circuit = buildCircuit(configuration)
        else:
            ansatz_circuit = self.ansatz_circuit
        return self.calculateEnergyValue(ansatz_circuit, shotsNum=shotsNum)

    def calculateEnergyValue(self, ansatz_circuit, shotsNum=1024):
        energy = 0.0
        for pauli_string, qubits, coeff in self.hamiltonian.to_sparse_list():
            ind_circuit = ansatz_circuit.copy()
            for qubit in qubits:
                # Measure the apportation to the energy for this Pauli string
                pauli = pauli_string[qubits.index(qubit)]
                if pauli == 'X':
                    hadamard_gate = {
                        "gate": "H",
                        "qubit": {"Number": qubit}
                    }
                    addGate(ind_circuit, hadamard_gate)
                elif pauli == 'Y':
                    ry_gate = {
                        "gate": "RY",
                        "qubit": {"Number": qubit},
                        "angle": -np.pi/2
                    }
                    hadamard_gate = {
                        "gate": "H",
                        "qubit": {"Number": qubit}
                    }
                    addGate(ind_circuit, ry_gate)
                    addGate(ind_circuit, hadamard_gate)
                
                if qubit == qubits[0]: ind_circuit.barrier()
                # For 'Z' and 'I', no basis change is needed
                measure_gate = {
                    "gate": "Measure",
                    "qubit": {"Number": qubit},
                    "classicalBit": qubit
                }
                addGate(ind_circuit, measure_gate)
            job = self.backend.run(ind_circuit, shots=shotsNum)
            counts = job.result().get_counts()
            
            energy += sigma_expectation(counts, qubits) * coeff

        return energy

    
    def minimizeEnergy(self, initial_angles, shotsNum=1024):
        from scipy.optimize import minimize

        result = minimize(self.getEnergy,
                          initial_angles,
                          args=(shotsNum),
                          method='COBYLA')
        return result

def build_simple_Ising_hamiltonian(numQubits, J, h_field):
    '''
    Builds the transverse Ising Hamiltonian:
    H = - \sum Z_i Z_{i+1} - h \sum X_i
    for a chain of numQubits qubits with transverse field h_field.
    Returns a SparsePauliOp representing the Hamiltonian.
    '''
    ham_list = []
    baseMatrix = ["I"] * numQubits
    for qNum in range(1,numQubits+1):
        # Interaction term
        if qNum < numQubits:
            transverseTerm = baseMatrix.copy()
            transverseTerm[qNum-1]   = "Z"
            transverseTerm[qNum+1-1] = "Z"
            ham_list += [("".join(transverseTerm), -J)]
        # Transverse field term
        fieldTerm = baseMatrix.copy()
        fieldTerm[qNum-1] = "X"
        ham_list += [("".join(fieldTerm), -h_field)]
        
    hamiltonian = SparsePauliOp.from_list(ham_list)
    return hamiltonian

def VQE_Energy_Field(numQubits, J, h_values, shotsNum=1024,
                     algorithmRuns=1, validateRuns=True,
                     plot=True, compareNumerical=False):
    energy_means = []
    energy_stds  = []
    energy_mins  = []
    energy_values = []
    if compareNumerical: num_energy_values = []
    for h_field in h_values:
        hamiltonian = build_simple_Ising_hamiltonian(numQubits, J, h_field)

        h_energy_values = []
        h_angle_values = []
        for _ in range(algorithmRuns):
            solver = VQESolver(numQubits, hamiltonian)
            initial_angles = [0.1, 0.2] + [0.3]*numQubits
            result = solver.minimizeEnergy(initial_angles, shotsNum=shotsNum)
            h_energy_values += [result.fun]
            h_angle_values  += [result.x]

        # Statistics over runs
        energy_mean = float(np.mean(h_energy_values))
        energy_std  = float(np.std(h_energy_values, ddof=0))
        energy_min  = float(np.min(h_energy_values))
        best_idx    = int(np.argmin(h_energy_values))
        best_angles = h_angle_values[best_idx]

        energy_means += [energy_mean]
        energy_stds  += [energy_std]
        energy_mins  += [energy_min]

        # validate best result with more shots to reduce shot-noise bias
        if validateRuns:
            validator = VQESolver(numQubits, hamiltonian)
            shots_validate = max(shotsNum * 10, 8192)
            best_energy_validated = float(validator.getEnergy(best_angles, shotsNum=shots_validate))
            energy_value = best_energy_validated
        else:
            energy_value = energy_mean

        # Save energy
        energy_values += [energy_value]

        #print("Optimal angles:", result.x)
        #print("Minimum energy:", result.fun)

        if compareNumerical:
            eigVals, eigVecs = np.linalg.eig(hamiltonian.to_matrix())
            eigVals, eigVecs = sortEigenstates(eigVals, eigVecs)
            num_energy_value = np.min(eigVals).astype(float)
            num_energy_values  += [num_energy_value]
            print("h", h_field, "Deviation:", 1 - energy_value / num_energy_value)
    
    if plot:
        if False:
            import matplotlib.pyplot as plt 
            #plt.plot(h_values, energy_values, marker='o', label="Simulation")
            # plot mean with errorbars and validated best points
            plt.errorbar(h_values, energy_means, yerr=energy_stds, marker='o', capsize=3, label="Simulations $\mu ± \sigma$")
            plt.plot(h_values, energy_values, marker='x', linestyle='None', color='C1', label="Best Simulation")
            if compareNumerical:
                plt.plot(h_values, num_energy_values, marker='o', label="Numerical")
            plt.xlabel("Transverse field $h$")
            plt.ylabel("Ground state energy (VQE)")
            plt.title(f"VQE Ground State Energy for Transverse Ising Model ({numQubits} qubits, {shotsNum} shots)")
            plt.grid()  
            plt.legend()
            plt.show()
        import matplotlib.pyplot as plt 
        # Darker palette
        color_num = "#0B3D91"   # dark blue
        color_best = "#8B0000"  # dark red
        color_sim  = "#006400"  # dark green

        # Plot simulations (mean ± std) first so Best Simulation is drawn on top
        plt.errorbar(h_values, energy_means,
                     yerr=energy_stds,
                     fmt='s', markersize=4, markeredgewidth=0.6,
                     ecolor=color_sim, color=color_sim,
                     capsize=3, elinewidth=2.0,
                     linewidth=0.8,
                     label="Simulations $\mu \\pm \\sigma$",
                     zorder=1)

        # Best (validated) simulation: plotted on top (superposed)
        plt.plot(h_values, energy_values,
                 linestyle='-', marker='s', markersize=4, markeredgewidth=0.6,
                 color=color_best, label="Best Simulation",
                 linewidth=0.8, zorder=3)

        if compareNumerical:
            # Numerical reference (dark blue), slimmer line and smaller markers
            plt.plot(h_values, num_energy_values,
                     marker='o', markersize=4, markeredgewidth=0.6,
                     color=color_num, label="Numerical",
                     linewidth=0.8, zorder=2)

        plt.xlabel("Transverse field $h$")
        plt.ylabel("Ground state energy (VQE)")
        plt.title(f"VQE Ground State Energy for Transverse Ising Model ({numQubits} qubits, {shotsNum} shots)")
        plt.grid()
        plt.legend()
        plt.show()        
    
    return energy_values

def VQE_Energy_ComputationTime(numQubits, J, h_field):
    import time
    hamiltonian = build_simple_Ising_hamiltonian(numQubits, J, h_field)

    solver = VQESolver(numQubits, hamiltonian)
    start_time = time.time()
    initial_angles = [0.1, 0.2] + [0.3]*numQubits
    result = solver.minimizeEnergy(initial_angles)
    end_time = time.time()
    computation_time = end_time - start_time
    print("Optimal angles:", result.x)
    print("Minimum energy:", result.fun)
    print(f"Computation time for {numQubits} qubits: {computation_time:.4f} seconds")
    return result.fun, computation_time

if __name__ == "__main__":
    # H = - J \sum Z_i Z_{i+1} - h \sum X_i
    J = 1
    numQubits = 3
    h_field = 0.5
    h_values = np.linspace(0.0, 2.0, 30)

    shots = 1024 * 2# Shots at each measure of part of the hamiltonian
    algorithmRuns = 5

    compareNumerical = True

    if True:
        energy_values = VQE_Energy_Field(numQubits, J, h_values,
                                         shotsNum=shots, algorithmRuns=algorithmRuns,
                                         plot=True, compareNumerical=compareNumerical)
    
    if False:
        numQubitsValues = [4, 6, 8, 10, 15, 20]
        compTimeValues = []
        for numQubits in numQubitsValues:
            energy, comp_time = VQE_Energy_ComputationTime(numQubits, J, h_field)
            compTimeValues += [comp_time]
        import matplotlib.pyplot as plt
        plt.plot(numQubitsValues, compTimeValues, marker='o')
        plt.xlabel("Number of Qubits")
        plt.ylabel("Computation Time (s)")


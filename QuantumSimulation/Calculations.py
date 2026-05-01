'''
Module with functions to calculate observables,
as well as generic operations, such as expectation values, fidelity and amplitude between states.
'''

from Utils import getTimer
from Operators import gauss_operator, buildChargeOperatorMinimal, buildPairCreationOperators, measure_electric_field
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.quantumcircuit import QuantumCircuit, ClassicalRegister
from qiskit.primitives import BaseEstimatorV2, BaseSamplerV2
import numpy as np

def checkChargeSymmetry(H, e0=0):
    '''
    Check if hamiltonian respects charge symmetry.

    Parameters:
    - H: Hamiltonian as a SparsePauliOp
    - e0: background electric field (default 0)

    Returns:
    - True if H respects charge symmetry, False otherwise
    - Charge operator as a SparsePauliOp
    '''
    # Length of the chain
    L = len(H.input_dims())

    # Q_n = e0 + 1/2 * sum_{m=0..n} (σ^z_m + (-1)^m)
    # Q_op = Sum(q_n)
    Q_op = buildChargeOperatorMinimal(L)

    # Conmuting check: [H, Q] = H*Q - Q*H
    commutator = (H.dot(Q_op) - Q_op.dot(H)).simplify()

    coeffs = np.asarray(commutator.coeffs).ravel()
    norm_commutator = np.sqrt((2 ** L) * np.sum(np.abs(coeffs) ** 2))

    print(f"{getTimer()} INFO: Norm of the conmutator: [H, Q] = {norm_commutator:.2e}")

    # If norm of the conmutator is close to zero, it commutes.
    if norm_commutator < 1e-9:
        print(f"{getTimer()} INFO: Hamiltonian respects charge symmetry.")
        return True, Q_op
    else:
        print(f"{getTimer()} WARNING: Hamiltonian does not respect charge symmetry.")
        return False, Q_op

def calculateOperatorExpectation(
        state: Statevector | QuantumCircuit,
        operator: SparsePauliOp,
        estimator: BaseEstimatorV2 | None = None,
        precision: float | None = None) -> float:
    '''
    Calculate the expectation value of an operator for a given state.

    '''
    if estimator is None:
        if isinstance(state, QuantumCircuit):
            state = Statevector(state)        
        return float(state.expectation_value(operator).real)
    else:
        assert isinstance(state, QuantumCircuit), "Estimator provided but state is already a Statevector. If estimator is provided, state must be a QuantumCircuit."
        if precision is not None:
            pub = [(state, operator, [], precision)]
        else:
            pub = [(state, operator)]
        job = estimator.run(pub)
        result = job.result()[0]
        return float(np.squeeze(result.data.evs).real)

def calculateAmplitude(
        state: QuantumCircuit | Statevector,
        target_state: QuantumCircuit | Statevector,
        sampler: BaseSamplerV2 | None = None
    ) -> complex:
    '''
    Calculate probability amplitude <target_state | state> as complex number.
    
    If Sampler is given, the interferometry Hadamard test is applied,
    using one ancilla qubit to extract amplitude's real and imaginary part.
    '''
    if sampler is None:
        if isinstance(state, QuantumCircuit):
            state = Statevector(state)
        if isinstance(target_state, QuantumCircuit):
            target_state = Statevector(target_state)
            
        return target_state.inner(state)
        
    else:
        # Hardware method: Hadamard test
        assert isinstance(state, QuantumCircuit), "State must be a QuantumCircuit."
        assert isinstance(target_state, QuantumCircuit), "Target state must be a QuantumCircuit."
        
        n_qubits = state.num_qubits
        
        # System is in state  |0 ... 0>.
        # 1. We create operator W = U_target^-1 * U_state
        # We want to measure <0| W |0> = <0| U_target^-1 * U_state |0> = <target|state>
        # with <0| W |0> = a + bi = z
        W = QuantumCircuit(n_qubits)
        W.compose(state, inplace=True)
        W.compose(target_state.inverse(), inplace=True)
        
        # 2. Convert W in a controlled operation
        cW = W.to_gate(label="cW").control(1)
        
        # 3. Create 1 qubit classical register to measure ancilla
        # We use ancilla at the begining and apply Hadamard such that
        # |state> = |0> \otimes |0 ... 0> ->
        # |state> = \frac{1}{\sqrt{2}} (|0> + |1>) \otimes |0 ... 0>
        cr = ClassicalRegister(1, "meas")
        
        # --- CIRCUIT FOR REAL PART ---
        # Qubit 0 will be ancilla, Qubits 1 a N will be logical system
        qc_real = QuantumCircuit(n_qubits + 1)
        # |state> = |0> \otimes |0 ... 0>
        qc_real.add_register(cr)
        # H|state> = H|0> \otimes |0 ... 0> = \frac{1}{\sqrt{2}} (|0> + |1>) \otimes |0 ... 0>
        qc_real.h(0) 
        # Apply controlled-W gate, with ancilla as control and system as target
        # (C-W) H |state> = \frac{1}{\sqrt{2}} (|0> \otimes |0 ... 0> + |1> \otimes W |0 ... 0>) 
        qc_real.append(cW, range(n_qubits + 1))
        # Apply another Hadamard on ancilla, since H \otimes H = 1
        qc_real.h(0)
        # H (C-W) H |state> = \frac{1}{\sqrt{2}} (H|0> \otimes |0 ... 0> + H|1> \otimes W |0 ... 0>) 
        # H (C-W) H |state> = \frac{1}{2} [ |0> \otimes ((1 + W) |0 ... 0>) + |1> \otimes ((1 - W) |0 ... 0>) ]
        # We measure P(0_A) = \frac{1}{4} | (1 + W) |0 ... 0> |^2
        # = \frac{1}{4} (<s|s> + <s|W|s> + <s|W^+|s> + <s|s>) = 1/4 (1 + z + z* + 1) = 1/2 (1 + Re(z)) =  1/2 (1 + a)
        # So, the real part of the measurement will be
        # a = 2 P(0_A) - 1
        qc_real.measure(0, 0)
        
        # --- CIRCUIT FOR IMAGINARY ---
        # It's analogous to the real part, except we insert an S^\dagger
        # so the imaginary part is extracted from the measurement (and not the real)
        qc_imag = QuantumCircuit(n_qubits + 1)
        qc_imag.add_register(cr)
        qc_imag.h(0)
        qc_imag.sdg(0) # Extra phase -pi/2 changes measurement basis
        qc_imag.append(cW, range(n_qubits + 1))
        qc_imag.h(0)
        qc_imag.measure(0, 0)
        
        # 4. Send both circuit to sampler in the same job
        job = sampler.run([qc_real, qc_imag])
        result = job.result()
        
        # 5. Extract probabilities to measure 0 on ancilla
        # Real part
        counts_real = result[0].data.meas.get_counts()
        shots_real = sum(counts_real.values())
        p0_real = counts_real.get('0', 0) / shots_real
        real_part = 2 * p0_real - 1
        
        # Imaginary part
        counts_imag = result[1].data.meas.get_counts()
        shots_imag = sum(counts_imag.values())
        p0_imag = counts_imag.get('0', 0) / shots_imag
        imag_part = 2 * p0_imag - 1
        
        # construct complex number
        return complex(real_part, imag_part)

def calculateFidelity(
        state_1: Statevector | QuantumCircuit,
        state_2: Statevector | QuantumCircuit,
        sampler: BaseSamplerV2 | None = None
    ) -> float:
    '''
    Calculate fidelity of two given states.
    '''
    if sampler is None:
        if isinstance(state_1, QuantumCircuit):
            state_1 = Statevector(state_1)
        if isinstance(state_2, QuantumCircuit):
            state_2 = Statevector(state_2)
        return float(np.abs(state_1.inner(state_2)) ** 2)
    else:
        # Hardware method (Compute-Uncompute)
        # state contains preparation ansatz + Trotter evolution
        assert isinstance(state_1, QuantumCircuit), "State 1 must be a QuantumCircuit when using sampler."
        assert isinstance(state_2, QuantumCircuit), "State 2 must be a QuantumCircuit when using sampler."
        # New circuit for measurement
        measure_circuit = state_1.copy()

        # Undo initial state preparation (we apply ansatz inverse)
        # initial_state_circuit es el circuito que preparó el vacío
        measure_circuit.compose(state_2.inverse(), inplace=True)

        # Measure all qubits at the end
        measure_circuit.measure_all()

        # Send to Sampler
        job = sampler.run([measure_circuit])
        result = job.result()[0]

        # Counts (shots) of the measurement results at the end
        counts = result.data.meas.get_counts()

        # Fidelity is the probability of measuring the all-zeros state
        # which would mean both states are the same.
        # So we take the count of the all-zeros state and divide by total shots.
        # If both states are the same (supposing vacuum), we would get
        # |00...0> -> Circuit|00...0> -> Circuit^-1 Circuit|00...0> -> |00...0>
        # so we would expect to measure all-zeros with probability 1, which means both states are the same, as expected.
        total_shots = sum(counts.values())
        zeros_state = '0' * measure_circuit.num_qubits

        fidelity = counts.get(zeros_state, 0) / total_shots   
        return float(fidelity)     

def calculateEnergy(
        state: Statevector | QuantumCircuit,
        hamiltonian: SparsePauliOp,
        estimator: BaseEstimatorV2 | None = None,
        precision: float | None = None
    ) -> float:
    '''
    Calculate energy as the expectation value of the Hamiltonian for a given state.

    '''
    return calculateOperatorExpectation(state, hamiltonian, estimator, precision)

def calculateVacuumPersistence(
        state: Statevector | QuantumCircuit,
        initial_state: Statevector | QuantumCircuit,
        sampler: BaseSamplerV2 | None = None
    ) -> float:
    '''
    Calculate vacuum persistence as the fidelity of a given state and the initial vacuum state.
    '''
    return calculateFidelity(state, initial_state, sampler)

def calculateGaussLawViolation(
        state: Statevector | QuantumCircuit,
        qubits_num: int,
        estimator: BaseEstimatorV2 | None = None,
        precision: float | None = None
    ) -> float:
    '''
    Check violation of Gauss' law as sum of the expectation value of the Gauss operator G_n
    of all the sites on the lattice. It should be 0 (or almost).
    '''
    value = 0
    for n in range(qubits_num):
        op = gauss_operator(n, qubits_num) @ gauss_operator(n, qubits_num)
        value += np.abs(calculateOperatorExpectation(state, op, estimator, precision))

    return value

def calculatePairCreation(
        state: Statevector | QuantumCircuit,
        qubits_num: int,
        estimator: BaseEstimatorV2 | None = None,
        precision: float | None = None
    ) -> tuple[float, float]:
    '''
    Calculate the number of pairs created as the sum of the occupation numbers of all sites.
    The occupation number of a site is calculated as n_occ = (1 + <Z>) / 2,
    where <Z> is the expectation value of the Z operator on that site.
    For even sites (electrons) we count the number of electrons created as 1 - n_occ,
    while for odd sites (positrons) we count the number of positrons created as n_occ.    
    '''
    n_e_obs, n_p_obs = buildPairCreationOperators(qubits_num)
    n_e = calculateOperatorExpectation(state, n_e_obs, estimator, precision)
    n_p = calculateOperatorExpectation(state, n_p_obs, estimator, precision)
    # Number of electrons and positrons
    return n_e, n_p

def calculateElectricField(
        state: Statevector | QuantumCircuit,
        qubits_num: int,
        e0: float = 0,
        estimator: BaseEstimatorV2 | None = None,
        precision: float | None = None
    ) -> np.array:
    '''
    Calculate the electric field at each link as E(n) = E_0 + sum_{k=0..n} Q_k, where Q_k is the charge operator at site k. Returns a list of the electric field at each link.
    '''
    #TODO: Implement efficiently and add estimator support
    electric_fields = measure_electric_field(state, qubits_num, e0)

    return electric_fields
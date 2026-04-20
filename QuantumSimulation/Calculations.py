from Utils import getTimer
from Operators import gauss_operator, buildChargeOperatorMinimal, buildPairCreationOperators
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.quantumcircuit import QuantumCircuit
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
        if isinstance(state, Statevector):
            raise TypeError("Estimator provided but state is already a Statevector. If estimator is provided, state must be a QuantumCircuit.")
        if precision is not None:
            pub = [(state, operator, [], precision)]
        else:
            pub = [(state, operator)]
        job = estimator.run(pub)
        result = job.result()[0]
        return float(np.squeeze(result.data.evs).real)

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
    Calculate vacuum persistence as the fidelity of a given state and the initial vacuum state
    '''
    if sampler is None:
        if isinstance(state, QuantumCircuit):
            state = Statevector(state)
        if isinstance(initial_state, QuantumCircuit):
            initial_state = Statevector(initial_state)
        return float(np.abs(state.inner(initial_state)) ** 2)
    else:
        # Hardware method (Compute-Uncompute)
        # state contains preparation ansatz + Trotter evolution
        assert isinstance(state, QuantumCircuit),         "State must be a QuantumCircuit when using sampler."
        assert isinstance(initial_state, QuantumCircuit), "Initial state must be a QuantumCircuit when using sampler."
        # New circuit for measurement
        measure_circuit = state.copy()

        # Undo initial state preparation (we apply ansatz inverse)
        # initial_state_circuit es el circuito que preparó el vacío
        measure_circuit.compose(initial_state.inverse(), inplace=True)

        # Measure all qubits at the end
        measure_circuit.measure_all()

        # Send to Sampler
        job = sampler.run([measure_circuit])
        result = job.result()[0]

        # Counts (shots) of the measurement results at the end
        counts = result.data.meas.get_counts()

        # Persistence is the probability of measuring the all-zeros state, which corresponds to the initial vacuum state after undoing the preparation.
        # So we take the count of the all-zeros state and divide by total shots.
        # If we are at initial state, we would get
        # |00...0> -> Circuit|00...0> -> Circuit^-1 Circuit|00...0> -> |00...0>
        # so we would expect to measure all-zeros with probability 1, which means persistence = 1 at t=0, as expected.
        total_shots = sum(counts.values())
        zeros_state = '0' * measure_circuit.num_qubits

        persistence = counts.get(zeros_state, 0) / total_shots   
        return float(persistence)     

def calculateGaussLawViolation(
        state: Statevector | QuantumCircuit,
        qubits_num: int,
        estimator: BaseEstimatorV2 | None = None,
        precision: float | None = None
    ) -> float:
    '''
    Check violation of Gauss' law as sum of the expectation value of the Gauss operator G_n of all the sites on the lattice. It should be 0 (or almost).
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
    Calculate the number of pairs created as the sum of the occupation numbers of all sites. The occupation number of a site is calculated as n_occ = (1 + <Z>) / 2, where <Z> is the expectation value of the Z operator on that site. For even sites (electrons) we count the number of electrons created as 1 - n_occ, while for odd sites (positrons) we count the number of positrons created as n_occ.    
    '''
    n_e_obs, n_p_obs = buildPairCreationOperators(qubits_num)
    n_e = calculateOperatorExpectation(state, n_e_obs, estimator, precision)
    n_p = calculateOperatorExpectation(state, n_p_obs, estimator, precision)
    # Number of electrons and positrons
    return n_e, n_p

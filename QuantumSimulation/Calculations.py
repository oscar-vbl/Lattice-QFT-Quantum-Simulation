from Utils import getTimer
from Operators import gauss_operator, buildChargeOperatorMinimal
from qiskit.quantum_info import SparsePauliOp, Statevector
import numpy as np

# --- comprobacion simetria de carga
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

    # Comprobamos la conmutación: [H, Q] = H*Q - Q*H
    commutator = (H.dot(Q_op) - Q_op.dot(H)).simplify()

    # Si la norma del conmutador es cercana a cero, conmuta.
    coeffs = np.asarray(commutator.coeffs).ravel()
    norm_commutator = np.sqrt((2 ** L) * np.sum(np.abs(coeffs) ** 2))

    print(f"{getTimer()} INFO: Norm of the conmutator: [H, Q] = {norm_commutator:.2e}")

    if norm_commutator < 1e-9:
        print(f"{getTimer()} INFO: Hamiltonian respects charge symmetry.")
        return True, Q_op
    else:
        print(f"{getTimer()} WARNING: Hamiltonian does not respect charge symmetry.")
        return False, Q_op

def calculateEnergy(state: Statevector, hamiltonian: SparsePauliOp):
    '''
    Calculate energy as the expectation value of the Hamiltonian for a given state.

    '''
    return state.expectation_value(hamiltonian).real

def calculateVaccumPersistance(state: Statevector, initial_state: Statevector):
    '''
    Calculate vaccum persistance as the fidelity of a given state and the initial vaccum state
    '''
    if initial_state is None:
        print(f"{getTimer()} WARNING: Initial state must be provided to calculate Persistance.")
        return None
    else:
        return np.abs(state.inner(initial_state)) ** 2

def calculateGaussLawViolation(state: Statevector, qubits_num: int):
    '''
    Check violation of Gauss' law as sum of the expectation value of the Gauss operator G_n of all the sites on the lattice. It should be 0 (or almost).
    '''
    value = 0
    for n in range(qubits_num):
        value += np.abs(state.expectation_value(gauss_operator(n, qubits_num) @ gauss_operator(n, qubits_num)))

    return value

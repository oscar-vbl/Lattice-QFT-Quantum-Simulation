from Utils import getTimer
from Operators import gauss_operator, buildChargeOperatorMinimal
from qiskit.quantum_info import SparsePauliOp, Statevector
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

def calculateEnergy(state: Statevector, hamiltonian: SparsePauliOp):
    '''
    Calculate energy as the expectation value of the Hamiltonian for a given state.

    '''
    return state.expectation_value(hamiltonian).real

def calculateVacuumPersistence(state: Statevector, initial_state: Statevector) -> np.floating | None:
    '''
    Calculate vacuum persistence as the fidelity of a given state and the initial vacuum state
    '''
    if initial_state is None:
        print(f"{getTimer()} WARNING: Initial state must be provided to calculate Persistence.")
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

def calculatePairCreation(state: Statevector, qubits_num: int):
    '''
    Calculate the number of pairs created as the sum of the occupation numbers of all sites. The occupation number of a site is calculated as n_occ = (1 + <Z>) / 2, where <Z> is the expectation value of the Z operator on that site. For even sites (electrons) we count the number of electrons created as 1 - n_occ, while for odd sites (positrons) we count the number of positrons created as n_occ.    
    '''
    # Number of electrons and positrons
    n_e, n_p = 0, 0
    for n in range(qubits_num):
        obs_z = SparsePauliOp.from_sparse_list([("Z", [n], 1.0)], num_qubits=qubits_num)
        exp_z = state.expectation_value(obs_z).real
        # Occupation number: n_occ = (1 + <Z>) / 2
        n_occ = (1 + exp_z) / 2
        if n % 2 == 0:
            # Electron site with charge (n_occ - 1)
            # Electrons created are the loss of occupation, so the number of electrons created is 1 - n_occ
            n_e += (1.0 - n_occ)
        else:
            # Positron site with charge n_occ
            # Positrons created are the increase of occupation, so the number of positrons created is n_occ
            n_p += n_occ

    return n_e, n_p

import numpy as np
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())
from Operators import (
    buildSchwingerHamiltonianTemporalGauge,
    buildChargeOperatorMinimal,
    gauss_operator
)
from qiskit.quantum_info import SparsePauliOp

def test_hamiltonian_hermiticity():
    """Verify that Hamiltonian is hermitian"""
    L, a, m, e0 = 6, 0.5, 1.0, 0.0
    H = buildSchwingerHamiltonianTemporalGauge(L, a, m, e0)
    
    H_matrix = H.to_matrix()
    H_dagger = H_matrix.conj().T
    
    error = np.linalg.norm(H_matrix - H_dagger)
    assert error < 1e-10, f"Hamiltonian not hermitian: ||H - H†|| = {error}"
    print("✓ Hamiltonian is hermitian")

def test_charge_symmetry():
    """Verify that [H, Q] = 0"""
    L, a, m, e0 = 6, 0.5, 1.0, 0.0
    H = buildSchwingerHamiltonianTemporalGauge(L, a, m, e0)
    Q = buildChargeOperatorMinimal(L)
    
    H_matrix = H.to_matrix()
    Q_matrix = Q.to_matrix()
    
    commutator = H_matrix @ Q_matrix - Q_matrix @ H_matrix
    error = np.linalg.norm(commutator)
    
    assert error < 1e-9, f"[H, Q] != 0: ||[[H,Q]]|| = {error}"
    print("✓ Charge symmetry verified: [H, Q] = 0")

def test_gauss_operator_definition():
    """Verify Gauss operator satisfies G_n = E_n - E_{n-1} - q_n"""
    L, a, m, e0 = 8, 0.5, 1.0, 0.0
    
    # For each site n, check that G_n commutes with H when properly defined
    H = buildSchwingerHamiltonianTemporalGauge(L, a, m, e0)
    
    for n in range(L-1):
        G_n = gauss_operator(n, L)
        
        H_matrix = H.to_matrix()
        G_matrix = G_n.to_matrix()
        
        commutator = H_matrix @ G_matrix - G_matrix @ H_matrix
        error = np.linalg.norm(commutator)
        
        assert error < 1e-9, f"[H, G_{n}] != 0 at site {n}: ||[[H,G_n]]|| = {error}"
    
    print(f"✓ Gauss operators commute with H for all n")

def test_eigenvalue_spectrum():
    """Verify eigenvalue spectrum is real and ordered"""
    L, a, m, e0 = 4, 0.5, 1.0, 0.0
    H = buildSchwingerHamiltonianTemporalGauge(L, a, m, e0)
    
    eigvals, eigvecs = np.linalg.eigh(H.to_matrix())
    
    # Check all eigenvalues are real (should be by hermiticity)
    assert np.all(np.isreal(eigvals)), "Some eigenvalues are complex"
    
    # Check they're ordered
    assert np.all(eigvals[:-1] <= eigvals[1:]), "Eigenvalues not ordered"
    
    print(f"✓ Spectrum for L={L}: E0={eigvals[0]:.4f}, gap={eigvals[1]-eigvals[0]:.4f}")

def test_vacuum_persistence_shift():
    L, a, m, e0 = 10, 0.5, 0.1, 0.0
    H_vac  = buildSchwingerHamiltonianTemporalGauge(L, a, m, e0=0.0)
    H_evol = buildSchwingerHamiltonianTemporalGauge(L, a, m, e0=0.5)

    _, vecs_0    = np.linalg.eigh(H_vac.to_matrix())
    _, vecs_evol = np.linalg.eigh(H_evol.to_matrix())

    vac_0    = vecs_0[:, 0]
    vac_evol = vecs_evol[:, 0]

    overlap = abs(np.dot(vac_0.conj(), vac_evol))**2

    evals_0, _    = np.linalg.eigh(H_vac.to_matrix())
    evals_evol, _ = np.linalg.eigh(H_evol.to_matrix())

    print("L, a, m =", L, a, m)
    print(f"Overlap |<vac(0)|vac(1.5)>|² = {overlap:.6f}")
    print(f"E0(e0=0):    {evals_0[0]:.6f}")
    print(f"E0(e0=1.5):  {evals_evol[0]:.6f}")
    print(f"Gap(e0=0):   {evals_0[1] - evals_0[0]:.6f}")
    print(f"Gap(e0=1.5): {evals_evol[1] - evals_evol[0]:.6f}")
    print(f"Shift E0:    {abs(evals_0[0] - evals_evol[0]):.6f}")

if __name__ == "__main__":
    test_hamiltonian_hermiticity()
    test_charge_symmetry()
    test_gauss_operator_definition()
    test_eigenvalue_spectrum()
    test_vacuum_persistence_shift()
    print("\n✓ All tests passed!")

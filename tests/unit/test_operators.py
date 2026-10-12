import numpy as np
import pytest

from QuantumSimulation.hamiltonians import SchwingerHamiltonianTemporalGauge
from QuantumSimulation.operators import (
    TotalChargeOperator, PairCreationOperator, ElectricFieldOperator,
    GaussOperator, get_local_charge_terms, get_local_number_terms,
    get_local_gauss_operator,
)


def test_local_charge_and_number_terms():
    assert get_local_charge_terms(0) == [("Z", [0], 0.5), ("I", [], 0.5)]
    assert get_local_charge_terms(1) == [("Z", [1], 0.5), ("I", [], -0.5)]
    assert get_local_number_terms(1) == [("Z", [1], 0.5), ("I", [], 0.5)]


def test_operator_component_counts(params4):
    assert TotalChargeOperator(params4).get_labels() == ["total"]
    assert PairCreationOperator(params4).get_labels() == ["electrons", "positrons"]
    assert ElectricFieldOperator(params4).get_labels() == ["link_0", "link_1", "link_2"]
    assert GaussOperator(params4).get_labels() == ["site_0", "site_1", "site_2", "site_3"]


def test_every_operator_has_correct_dimension(params4):
    operators = [
        TotalChargeOperator(params4), PairCreationOperator(params4),
        ElectricFieldOperator(params4), GaussOperator(params4),
        SchwingerHamiltonianTemporalGauge(params4),
    ]
    for operator in operators:
        for component in operator.to_qiskit():
            assert component.num_qubits == params4.L
            assert component.to_matrix().shape == (2**params4.L, 2**params4.L)


def test_hamiltonian_is_hermitian_and_charge_conserving(params4):
    h = SchwingerHamiltonianTemporalGauge(params4).to_qiskit(sparse=True).to_matrix()
    q = TotalChargeOperator(params4).to_qiskit(sparse=True).to_matrix()
    np.testing.assert_allclose(h, h.conj().T, atol=1e-12)
    np.testing.assert_allclose(h @ q - q @ h, 0.0, atol=1e-10)


def test_hamiltonian_term_counts(params4):
    h = SchwingerHamiltonianTemporalGauge(params4)
    assert len(h.terms_dict["hopping"]) == 2 * (params4.L - 1)
    assert len(h.terms_dict["mass"]) == params4.L
    assert "electric" in h.terms_dict


def test_periodic_boundary_rejected(params4):
    params4.periodic = True
    with pytest.raises(NotImplementedError):
        SchwingerHamiltonianTemporalGauge(params4)


def test_copy_weight_add_and_simplify(params2):
    original = TotalChargeOperator(params2)
    copied = original.copy()
    copied.update_with_weight(2.0)
    assert copied.terms_dict != original.terms_dict
    original.add_terms({"extra": [("Z", [0], 1.0), ("Z", [0], -1.0)]})
    original.simplify()
    assert original.terms_dict["extra"] == []


def test_local_gauss_operator_is_identically_zero_after_simplification(params4):
    # With E_i defined from cumulative charge, E_i-E_{i-1}-q_i is algebraically zero.
    for i in range(params4.L):
        terms = get_local_gauss_operator(i)
        from qiskit.quantum_info import SparsePauliOp
        op = SparsePauliOp.from_sparse_list(terms, num_qubits=params4.L).simplify()
        np.testing.assert_allclose(op.to_matrix(), 0.0, atol=1e-12)

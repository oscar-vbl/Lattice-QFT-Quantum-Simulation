import numpy as np
import pytest
from qiskit import QuantumCircuit

from QuantumSimulation.hamiltonians import SchwingerHamiltonianTemporalGauge
from QuantumSimulation.observables import (
    EnergyObservable, ElectricFieldObservable, GaussLawViolationObservable,
    PairCreationObservable, PersistenceObservable,
)
from QuantumSimulation.observables.charge import TotalChargeObservable
from QuantumSimulation.operators import (
    TotalChargeOperator, ElectricFieldOperator, GaussOperator, PairCreationOperator,
)


def test_energy_exact_matches_direct_matrix(params2, bell_2):
    h = SchwingerHamiltonianTemporalGauge(params2)
    obs = EnergyObservable(h)
    value, error = obs.calculate_exact(bell_2)
    sv = __import__("qiskit").quantum_info.Statevector(bell_2)
    expected = np.real(np.vdot(sv.data, h.to_qiskit(sparse=True).to_matrix() @ sv.data))
    assert value == pytest.approx(expected)
    assert error == 0.0


def test_charge_on_staggered_vacuum_is_zero(params2, staggered_2):
    value, error = TotalChargeObservable(TotalChargeOperator(params2)).calculate_exact(staggered_2)
    assert value == pytest.approx(0.0)
    assert error == 0.0


def test_pair_creation_on_reference_vacuum_is_zero(params2, staggered_2):
    (electrons, positrons), errors = PairCreationObservable(
        PairCreationOperator(params2)
    ).calculate_exact(staggered_2)
    assert electrons == pytest.approx(1.0)
    assert positrons == pytest.approx(1.0)
    assert errors == (0.0, 0.0)


def test_electric_field_has_l_minus_one_values(params4, staggered_2):
    qc = QuantumCircuit(4)
    qc.x(0)
    qc.x(2)
    values, errors = ElectricFieldObservable(ElectricFieldOperator(params4)).calculate_exact(qc)
    assert values.shape == (params4.L - 1,)
    assert errors.shape == values.shape
    np.testing.assert_allclose(errors, 0.0)


def test_gauss_violation_zero_by_construction(params4):
    qc = QuantumCircuit(params4.L)
    qc.x(0)
    qc.x(2)
    value, error = GaussLawViolationObservable(GaussOperator(params4)).calculate_exact(qc)
    assert value == pytest.approx(0.0, abs=1e-12)
    assert error == 0.0


def test_persistence_identical_and_orthogonal_states(zero_2):
    obs = PersistenceObservable(zero_2)
    same, err = obs.calculate_exact(zero_2.copy())
    orthogonal = zero_2.copy()
    orthogonal.x(0)
    other, _ = obs.calculate_exact(orthogonal)
    assert same == pytest.approx(1.0)
    assert other == pytest.approx(0.0)
    assert err == 0.0

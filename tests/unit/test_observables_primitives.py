import pytest
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

from QuantumSimulation.hamiltonians import SchwingerHamiltonianTemporalGauge
from QuantumSimulation.observables import EnergyObservable, PersistenceObservable


def test_energy_estimator_matches_exact(params2, bell_2):
    obs = EnergyObservable(SchwingerHamiltonianTemporalGauge(params2))
    exact, _ = obs.calculate_exact(bell_2)
    result = StatevectorEstimator().run([obs.get_pub(bell_2)]).result()
    estimated, error = obs.process_pub_result(result[0])
    assert estimated == pytest.approx(exact, abs=1e-10)
    assert error >= 0


def test_persistence_sampler_identical_state(staggered_2):
    obs = PersistenceObservable(staggered_2, shots=8192)
    result = StatevectorSampler(default_shots=8192).run([obs.get_pub(staggered_2)]).result()
    value, error = obs.process_pub_result(result[0])
    assert value == pytest.approx(1.0, abs=1e-12)
    assert error == pytest.approx(0.0, abs=1e-12)

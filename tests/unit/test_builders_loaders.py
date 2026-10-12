import pytest

from QuantumSimulation.core.loaders import load_evolution_observables
from QuantumSimulation.hamiltonians import build_hamiltonian, SchwingerHamiltonianTemporalGauge
from QuantumSimulation.core.types import HamiltonianConfig
from QuantumSimulation.observables._builder import build_observable, get_available_observables
from QuantumSimulation.operators._builder import build_operator, get_available_operators
from QuantumSimulation.operators import BaseOperator


def test_registered_operators_build(params2):
    for name in get_available_operators():
        assert isinstance(build_operator(name, params2), BaseOperator)
    with pytest.raises(ValueError):
        build_operator("missing", params2)


def test_observable_builder_requirements(params2, staggered_2):
    charge = build_operator("Total_Charge_Operator", params2)
    assert build_observable("Total_Charge", operator=charge).primitive_type == "estimator"
    assert build_observable("Persistence", reference_state=staggered_2).primitive_type == "sampler"
    with pytest.raises(ValueError):
        build_observable("Energy", operator=None)
    with pytest.raises(ValueError):
        build_observable("Persistence", reference_state=None)


def test_load_all_evolution_observables(params2, staggered_2):
    h = SchwingerHamiltonianTemporalGauge(params2)
    names = tuple(get_available_observables())
    loaded = load_evolution_observables(
        names, {}, params2, hamiltonian=h, reference_state=staggered_2,
        precision=None, shots=1024,
    )
    assert [o.name for o in loaded] == list(names)


def test_hamiltonian_builder_override_does_not_mutate(params2):
    cfg = HamiltonianConfig("Schwinger", "Temporal", params2)
    h = build_hamiltonian(cfg, {"e0": 0.7})
    assert h.params.e0 == 0.7
    assert params2.e0 == 0.0

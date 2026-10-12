import pytest
from qiskit.quantum_info import Statevector

from QuantumSimulation.ansatz.initial_state import build_initial_state
from QuantumSimulation.ansatz import build_parametrized_ansatz
from QuantumSimulation.core.types import InitialStateConfig, AnsatzConfig
from QuantumSimulation.hamiltonians import SchwingerHamiltonianTemporalGauge


def test_staggered_initial_state(params4):
    qc = build_initial_state(params4.L, InitialStateConfig())
    probs = Statevector(qc).probabilities_dict()
    assert len(probs) == 1
    # Qiskit bitstrings are shown in reverse qubit order; q0 and q2 are one.
    assert next(iter(probs.values())) == pytest.approx(1.0)
    assert qc.count_ops().get("x", 0) == 2


#@pytest.mark.parametrize("ansatz_type", ["EfficientSU2", "ExcitationPreserving", "TwoLocal"])
#def test_standard_ansatz_builds(params4, ansatz_type):
#    cfg = AnsatzConfig(ansatz_type, InitialStateConfig(), reps=1)
#    qc = build_parametrized_ansatz(params4.L, cfg)
#    assert qc.num_qubits == params4.L
#    assert qc.num_parameters > 0


@pytest.mark.parametrize("ansatz_type", ["HVA", "HVA Full", "HVA Simple"])
def test_hva_builds(params4, ansatz_type):
    cfg = AnsatzConfig(ansatz_type, InitialStateConfig(), reps=1)
    qc = build_parametrized_ansatz(
        params4.L, cfg, SchwingerHamiltonianTemporalGauge(params4)
    )
    assert qc.num_qubits == params4.L
    assert qc.num_parameters > 0


def test_hva_requires_hamiltonian(params4):
    cfg = AnsatzConfig("HVA", InitialStateConfig(), reps=1)
    with pytest.raises(ValueError):
        build_parametrized_ansatz(params4.L, cfg)

import numpy as np
from qiskit import QuantumCircuit

from QuantumSimulation.algorithms import VQESolver
from QuantumSimulation.core.types import OptimizerConfig, PrimitiveBundle
from QuantumSimulation.hamiltonians import SchwingerHamiltonianTemporalGauge
from QuantumSimulation.observables import EnergyObservable


def _solver(params2, strategy="zeros"):
    ansatz = QuantumCircuit(2)
    from qiskit.circuit import Parameter
    theta = Parameter("theta")
    ansatz.ry(theta, 0)
    h = SchwingerHamiltonianTemporalGauge(params2)
    return VQESolver(
        ansatz, EnergyObservable(h), h,
        OptimizerConfig(maxiter=2, init_strategy=strategy),
        PrimitiveBundle(None, None, None, None, None),
    )


def test_initial_parameter_strategies(params2):
    assert np.array_equal(_solver(params2, "zeros")._generate_initial_params(), [0.0])
    assert _solver(params2, "random_small")._generate_initial_params().shape == (1,)
    values = _solver(params2, "uniform_random")._generate_initial_params()
    assert np.all((values >= 0) & (values <= 2 * np.pi))


def test_exact_cost_and_short_solve(params2):
    solver = _solver(params2)
    assert np.isfinite(solver._energy_cost_function(np.array([0.0])))
    result = solver.solve(np.array([0.0]))
    assert np.isfinite(result.energy)
    assert result.parameters.shape == (1,)

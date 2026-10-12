import numpy as np
import pytest
from qiskit import QuantumCircuit

from QuantumSimulation.core.types import (
    EvolutionResult, TrajectoryResult, ObservableSeries,
    MitigationConfig, MitigationStepConfig,
)
from QuantumSimulation.mitigation import apply_mitigation_pipeline


def _trajectory(key, step, values, sigma=0.01):
    values = np.asarray(values, dtype=float)
    return TrajectoryResult(
        key=key,
        times=np.arange(values.shape[0], dtype=float),
        observables={"Energy": ObservableSeries(values, np.full_like(values, sigma))},
        final_state=QuantumCircuit(1),
        step_size=step,
    )


def test_richardson_recovers_second_order_limit():
    exact = np.array([1.0, 2.0, 3.0])
    coarse = exact + 0.8 * 0.2**2
    fine = exact + 0.8 * 0.1**2
    result = EvolutionResult(
        {"coarse": _trajectory("coarse", 0.2, coarse), "fine": _trajectory("fine", 0.1, fine)},
        "fine",
    )
    config = MitigationConfig(
        enabled=True,
        steps=(MitigationStepConfig("richardson", {
            "coarse_key": "coarse", "fine_key": "fine", "order": 2,
            "output_key": "richardson",
        }),),
        preferred_output="richardson",
    )
    out = apply_mitigation_pipeline(result, config)
    np.testing.assert_allclose(out.preferred.observables["Energy"].values, exact)
    assert out.preferred.final_state is None
    assert out.preferred.state_source_key == "fine"
    assert out.final_state is out.trajectories["fine"].final_state
    assert len(out.mitigation_history) == 1


def test_richardson_uncertainty_formula():
    result = EvolutionResult(
        {"c": _trajectory("c", 0.2, [1.0], 0.02), "f": _trajectory("f", 0.1, [1.0], 0.01)},
        "f",
    )
    config = MitigationConfig(True, (MitigationStepConfig("richardson", {
        "coarse_key": "c", "fine_key": "f", "order": 2, "output_key": "r"
    }),), "r")
    out = apply_mitigation_pipeline(result, config)
    expected = np.sqrt((4 / 3 * 0.01) ** 2 + (1 / 3 * 0.02) ** 2)
    assert out.preferred.observables["Energy"].uncertainties[0] == pytest.approx(expected)


def test_unknown_mitigation_rejected():
    result = EvolutionResult({"x": _trajectory("x", 0.1, [1.0])}, "x")
    config = MitigationConfig(True, (MitigationStepConfig("unknown"),), None)
    with pytest.raises(ValueError, match="Unknown"):
        apply_mitigation_pipeline(result, config)

import numpy as np
import pytest
from qiskit import QuantumCircuit

from QuantumSimulation.core.types import (
    EvolutionConfig, EvolutionVariantConfig, MitigationConfig,
    MitigationStepConfig, ObservableSeries, TrajectoryResult, EvolutionResult,
)


def test_evolution_config_grid_and_lookup():
    cfg = EvolutionConfig(
        active=True, total_time=1.0, time_steps=4,
        variants=(
            EvolutionVariantConfig("dt", "trotter", 1.0),
            EvolutionVariantConfig("dt_half", "trotter", 0.5),
        ),
        preferred_variant="dt_half",
    )
    assert cfg.base_step == pytest.approx(0.25)
    np.testing.assert_allclose(cfg.times, np.linspace(0, 1, 5))
    assert cfg.get_variant("dt_half").step_multiplier == 0.5


@pytest.mark.parametrize("multiplier", [0.3, 0.6, 0.7])
def test_invalid_trotter_multiplier_rejected(multiplier):
    with pytest.raises(ValueError, match="divide"):
        EvolutionConfig(
            active=True, total_time=1, time_steps=2,
            variants=(EvolutionVariantConfig("bad", "trotter", multiplier),),
        )


def test_duplicate_variants_rejected():
    with pytest.raises(ValueError, match="unique"):
        EvolutionConfig(
            active=True, total_time=1, time_steps=2,
            variants=(EvolutionVariantConfig("x"), EvolutionVariantConfig("x")),
        )


def test_observable_series_shapes():
    with pytest.raises(ValueError, match="same shape"):
        ObservableSeries(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="component labels"):
        ObservableSeries(np.zeros((3, 2)), np.zeros((3, 2)), ("x",))


def test_evolution_result_resolves_mitigated_state():
    state = QuantumCircuit(1)
    raw = TrajectoryResult("fine", np.array([0.0, 1.0]), {}, final_state=state)
    mitigated = TrajectoryResult(
        "richardson", np.array([0.0, 1.0]), {},
        final_state=None, state_source_key="fine",
    )
    result = EvolutionResult({"fine": raw, "richardson": mitigated}, "richardson")
    assert result.final_state is state


def test_mitigation_config_validation():
    with pytest.raises(ValueError):
        MitigationConfig(enabled=True, steps=())
    cfg = MitigationConfig(
        enabled=True,
        steps=(MitigationStepConfig("richardson", {"output_key": "r"}),),
        preferred_output="r",
    )
    assert cfg.preferred_output == "r"

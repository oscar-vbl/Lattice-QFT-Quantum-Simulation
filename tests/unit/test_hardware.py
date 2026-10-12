import pytest

from QuantumSimulation.core.types import HardwareConfig, NoiseConfig, NoiseErrorConfig
from QuantumSimulation.hardware import setup_quantum_primitives
from QuantumSimulation.hardware.noise import build_noise_model


def test_no_backend_bundle():
    bundle = setup_quantum_primitives(HardwareConfig())
    assert bundle.backend is None
    assert bundle.estimator is None
    assert bundle.sampler is None


def test_ideal_bundle():
    bundle = setup_quantum_primitives(HardwareConfig(backend_type="StatevectorEstimator", shots=128))
    assert bundle.estimator is not None
    assert bundle.sampler is not None


def test_unknown_backend_rejected():
    with pytest.raises(NotImplementedError):
        setup_quantum_primitives(HardwareConfig(backend_type="unknown"))


@pytest.mark.aer
def test_depolarizing_noise_model():
    cfg = NoiseConfig([NoiseErrorConfig("depolarizing", 1, 0.01, ["x"])])
    model = build_noise_model(cfg)
    assert model is not None
    assert "x" in model.noise_instructions


from QuantumSimulation.core.types import SymmetryCheckConfig
from QuantumSimulation.diagnostics import PhysicsValidator
from QuantumSimulation.hamiltonians import SchwingerHamiltonianTemporalGauge
from QuantumSimulation.observables.charge import TotalChargeObservable
from QuantumSimulation.operators import TotalChargeOperator


def test_charge_symmetry_verified(params2):
    h = SchwingerHamiltonianTemporalGauge(params2)
    q = TotalChargeOperator(params2)
    validator = PhysicsValidator([q])
    out = validator.enforce_symmetries(h, [SymmetryCheckConfig(q.name, tolerance=1e-10)])
    assert out is not h


def test_state_validation_pass_and_fail(params2, staggered_2, zero_2):
    obs = TotalChargeObservable(TotalChargeOperator(params2))
    validator = PhysicsValidator([obs])
    check = SymmetryCheckConfig("Total_Charge", target_value=0.0, tolerance=1e-10)
    assert validator.validate_state(staggered_2, [check])["Total_Charge"]["passed"]
    assert not validator.validate_state(zero_2, [check])["Total_Charge"]["passed"]

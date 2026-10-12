import pytest
from qiskit import QuantumCircuit

from QuantumSimulation.core.types import SchwingerParams


@pytest.fixture
def params2():
    return SchwingerParams(L=2, a=0.5, m=0.2, e0=0.0, periodic=False)


@pytest.fixture
def params4():
    return SchwingerParams(L=4, a=0.5, m=0.2, e0=0.1, periodic=False)


@pytest.fixture
def staggered_2():
    qc = QuantumCircuit(2)
    qc.x(0)
    return qc


@pytest.fixture
def zero_2():
    return QuantumCircuit(2)


@pytest.fixture
def bell_2():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def simple_config():
    return {
        "QubitsNumber": 2,
        "Hamiltonian": {
            "Type": "Schwinger",
            "Gauge": "Temporal",
            "Parameters": {"L": 2, "a": 0.5, "m": 0.2, "e0": 0.0},
        },
        "Backend": {"Type": "StatevectorEstimator", "Shots": 4096},
        "Ansatz": {
            "Type": "ExcitationPreserving",
            "Reps": 1,
            "Initial State": {"Vacuum": True, "Staggered": True},
            "Init_Strategy": "zeros",
            "Minimizer": {"Method": "COBYLA", "Options": {"maxiter": 2}},
        },
        "Diagnostics": {"Symmetries": [], "StateValidation": []},
        "Temporal Evolution": {"Active": False},
    }

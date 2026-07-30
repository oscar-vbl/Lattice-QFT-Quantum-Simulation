from .base import BaseObservable
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.primitives import (
    PubResult,
)
from typing import Iterable

class EnergyObservable(BaseObservable):
    """
    Calculate energy as the expectation value of the Hamiltonian for a given state.
    """

    def __init__(self, hamiltonian, precision=None):
        super().__init__("Energy", "estimator")
        self.hamiltonian = hamiltonian
        self.precision = precision

    def get_pub(self, state: QuantumCircuit) -> tuple:
        # PUB for Estimator
        return (state, self.hamiltonian, None, self.precision)

    def process_pub_result(self, result: PubResult) -> float:
        return self._pubOperatorExpectation(result)

    def get_operators(self) -> list[SparsePauliOp]:
        return [self.hamiltonian]

    def process_result_data(self, evs: Iterable, stds: Iterable) -> tuple[float, float]:
        return float(evs[0]), float(stds[0])

    def calculate_exact(self, state: Statevector | QuantumCircuit) -> float:
        return self._exactOperatorExpectation(state, self.hamiltonian)


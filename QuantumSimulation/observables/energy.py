from .base import BaseObservable
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.primitives import (
    PubResult,
)
from typing import Iterable
from QuantumSimulation.operators import BaseOperator

class EnergyObservable(BaseObservable):
    """
    Calculate energy as the expectation value of the Hamiltonian for a given state.
    """

    def __init__(self, operator: BaseOperator, precision=None):
        super().__init__("Energy", "estimator")
        self.operator = operator
        self.precision = precision

        self.hamiltonian_op = self.operator.to_qiskit(sparse=True)

    def get_pub(self, state: QuantumCircuit) -> tuple:
        # PUB for Estimator
        return (state, self.hamiltonian_op, None, self.precision)

    def process_pub_result(self, result: PubResult) -> float:
        return self._pubOperatorExpectation(result)

    def get_operators(self) -> list[SparsePauliOp]:
        return [self.hamiltonian_op]

    def process_result_data(self, evs: Iterable, stds: Iterable) -> tuple[float, float]:
        return float(evs[0]), float(stds[0])

    def calculate_exact(self, state: Statevector | QuantumCircuit) -> float:
        return self._exactOperatorExpectation(state, self.hamiltonian_op)


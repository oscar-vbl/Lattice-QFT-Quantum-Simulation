# QuantumSimulation/observables/fermions.py
from typing import Iterable
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.primitives import PubResult

from .base import BaseObservable
from QuantumSimulation.operators import BaseOperator

class TotalChargeObservable(BaseObservable):
    """
    Calculate the total charge of the lattice to validate symmetries (e.g., Q = 0).
    Receives an agnostic BaseOperator and evaluates it on a quantum state.
    """

    def __init__(self, operator: BaseOperator, precision: float = None):
        super().__init__("Total_Charge", "estimator")
        self.num_qubits = operator.num_qubits
        self.precision = precision

        self.charge_operator = operator
        
        # Compile to qiskit.
        self.charge_op = self.charge_operator.to_qiskit(sparse=True)

    def get_pub(self, state: QuantumCircuit) -> tuple:
        return (state, self.charge_op, None, self.precision)

    def get_operators(self) -> list[SparsePauliOp]:
        return [self.charge_op]

    def process_pub_result(self, result: PubResult) -> float:
        return self._pubOperatorExpectation(result)

    def process_result_data(self, evs: Iterable, stds: Iterable) -> tuple[float, float]:
        return float(evs[0]), float(stds[0])

    def calculate_exact(self, state: Statevector | QuantumCircuit) -> tuple[float, float]:
        return self._exactOperatorExpectation(state, self.charge_op)
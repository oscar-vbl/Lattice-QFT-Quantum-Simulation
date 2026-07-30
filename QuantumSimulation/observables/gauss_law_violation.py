from .base import BaseObservable
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.primitives import (
    PubResult,
)
import numpy as np
from QuantumSimulation.Operators import gauss_operator

class GaussLawViolationObservable(BaseObservable):
    """
    Check violation of Gauss' law as sum of the expectation value of the squared Gauss operator G_n^2
    of all the sites on the lattice.
    """

    def __init__(self, qubits_num: int, precision=None):
        super().__init__("Gauss_Law_Violation", "estimator")
        self.precision = precision
        self.qubits_num = qubits_num

        # Prebuild squared Gauss operators for all sites.
        self.gauss_squared_ops = []
        for n in range(qubits_num):
            op = gauss_operator(n, qubits_num) @ gauss_operator(n, qubits_num)
            self.gauss_squared_ops.append(op.simplify())

    def get_pub(self, state: QuantumCircuit) -> tuple:
        # Send complete list of operators.
        return (state, self.gauss_squared_ops, None, self.precision)

    def get_operators(self) -> list[SparsePauliOp]:
        return self.gauss_squared_ops

    def process_result_data(self, evs, stds) -> tuple[float, float]:
        evs_real  = np.squeeze(evs).real
        stds_real = np.squeeze(stds).real

        total_violation = float(np.sum(np.abs(evs_real)))
        # Square sum of errors, assuming they are independent, to get total error estimation.
        total_error = float(np.sqrt(np.sum(stds_real**2)))

        return total_violation, total_error

    def process_pub_result(self, result: PubResult) -> tuple[float, float]:
        # evs is a numpy array on length number qubits (G_n^2)
        evs = np.squeeze(result.data.evs).real
        stds = np.squeeze(result.data.stds).real

        # Sum all absolute values, to sum also local violations cancelled.
        total_violation = np.sum(np.abs(evs))
        total_error = np.sqrt(np.sum(stds**2))

        return float(total_violation), float(total_error)

    def calculate_exact(self, state: Statevector | QuantumCircuit) -> tuple:
        total_violation = 0.0
        for op in self.gauss_squared_ops:
            val = self._exactOperatorExpectation(state, op)
            total_violation += np.abs(val)

        return float(total_violation), 0.0


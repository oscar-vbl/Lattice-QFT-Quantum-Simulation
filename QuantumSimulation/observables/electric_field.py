from .base import BaseObservable
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.primitives import (
    PubResult,
)
import numpy as np
from QuantumSimulation.Operators import buildElectricFieldOperators

class ElectricFieldObservable(BaseObservable):
    """
    Calculate the electric field at each link as E(n) = E_0 + sum_{k=0..n} Q_k.
    Returns an array of L-1 values.
    """

    def __init__(self, qubits_num: int, e0: float = 0.0, precision=None):
        super().__init__("Electric_Field", "estimator")
        self.qubits_num = qubits_num
        self.e0 = e0
        self.precision = precision

        # Prebuild the list of L-1 operators (electric field at each lattice link)
        self.ef_ops = buildElectricFieldOperators(qubits_num, e0)

    def get_pub(self, state: QuantumCircuit) -> tuple:
        # Send all L-1 operators in one PUB
        return (state, self.ef_ops, None, self.precision)

    def get_operators(self) -> list[SparsePauliOp]:
        return self.ef_ops

    def process_result_data(
        self, evs, stds
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        evs_real = np.squeeze(evs).real
        stds_real = np.squeeze(stds).real

        return evs_real, stds_real

    def process_pub_result(self, result: PubResult) -> tuple[np.ndarray, np.ndarray]:
        # evs will be a numpy array with L-1 values (the field at each link)
        evs = np.squeeze(result.data.evs).real
        stds = np.squeeze(result.data.stds).real

        return evs, stds

    def calculate_exact(
        self, state: Statevector | QuantumCircuit
    ) -> tuple[np.ndarray, np.ndarray]:
        ef_values = []
        for op in self.ef_ops:
            val = self._exactOperatorExpectation(state, op)
            ef_values.append(val)

        return np.array(ef_values), np.array([0.0] * len(ef_values))


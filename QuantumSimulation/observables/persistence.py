from .base import BaseObservable
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.primitives import (
    SamplerPubResult,
)

class PersistenceObservable(BaseObservable):
    """
    Calculate vacuum persistence as the fidelity of a given state and the initial vacuum state.
    """

    def __init__(self, initial_state: Statevector | QuantumCircuit, precision=None):
        super().__init__("Persistence", "sampler")
        self.initial_state = initial_state
        self.precision = precision

    def get_pub(
        self, state: Statevector | QuantumCircuit
    ) -> tuple[QuantumCircuit, None]:
        # PUB for Sampler
        return (self._getFidelityPub(state, self.initial_state), None)

    def get_operators(self) -> list[SparsePauliOp]:
        pass

    def process_result_data(
        self, evs, stds
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        pass

    def process_pub_result(self, result: SamplerPubResult) -> tuple[float, float]:
        return self._pubFidelityCalculation(result)

    def calculate_exact(
        self, state: Statevector | QuantumCircuit
    ) -> tuple[float, float]:
        return self._exactFidelityCalculation(state, self.initial_state)

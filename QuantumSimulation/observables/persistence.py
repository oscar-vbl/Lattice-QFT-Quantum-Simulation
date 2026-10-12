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

    def __init__(self, reference_state: Statevector | QuantumCircuit, shots=None):
        super().__init__("Persistence", "sampler")
        self.reference_state = reference_state
        self.shots = shots

    def get_pub(
        self, state: Statevector | QuantumCircuit
    ) -> tuple[QuantumCircuit, None]:
        # PUB for Sampler
        return (self._getFidelityPub(state, self.reference_state), None, self.shots)

    def get_operators(self) -> list[SparsePauliOp]:
        # It's a sampler observable
        return []

    def process_result_data(
        self, evs, stds
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        raise ValueError("Persistence is processed from Sampler results")

    def process_pub_result(self, result: SamplerPubResult) -> tuple[float, float]:
        return self._pubFidelityCalculation(result)

    def calculate_exact(
        self, state: Statevector | QuantumCircuit
    ) -> tuple[float, float]:
        return self._exactFidelityCalculation(state, self.reference_state)

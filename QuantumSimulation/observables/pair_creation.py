from .base import BaseObservable
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.primitives import (
    PubResult,
)
import numpy as np
from QuantumSimulation.Operators import buildPairCreationOperators

class PairCreationObservable(BaseObservable):
    """
    Calculate the number of pairs created as the sum of the occupation numbers of all sites.
    The occupation number of a site is calculated as n_occ = (1 + <Z>) / 2,
    where <Z> is the expectation value of the Z operator on that site.
    For even sites (electrons) we count the number of electrons created as 1 - n_occ,
    while for odd sites (positrons) we count the number of positrons created as n_occ.
    """

    def __init__(self, qubits_num: int, precision=None):
        super().__init__("Pair_Creation", "estimator")
        self.precision = precision
        self.qubits_num = qubits_num

        # Prebuild number of electrons and positrons operators.
        self.n_e_obs, self.n_p_obs = buildPairCreationOperators(qubits_num)

    def get_pub(self, state: QuantumCircuit) -> tuple:
        # Send in one pub
        return (state, [self.n_e_obs, self.n_p_obs], None, self.precision)

    def process_pub_result(self, result: PubResult) -> tuple[tuple, tuple]:
        evs, errors = self._pubOperatorExpectationMultiple(result)
        return tuple([ev for ev in evs]), tuple([err for err in errors])

    def get_operators(self) -> list[SparsePauliOp]:
        return [self.n_e_obs, self.n_p_obs]

    def process_result_data(
        self, evs, stds
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        evs_real = np.squeeze(evs).real
        stds_real = np.squeeze(stds).real

        n_e, n_p = 0.0, 0.0
        err_e_sq, err_p_sq = 0.0, 0.0

        for i, (val, err) in enumerate(zip(evs_real, stds_real)):
            if i % 2 == 0:  # Positrons (even sites)
                n_p += 0.5 * (val + 1)
                err_p_sq += (0.5 * err) ** 2
            else:  # Electrons (odd sites)
                n_e += 0.5 * (1 - val)
                err_e_sq += (0.5 * err) ** 2

        return (n_e, n_p), (np.sqrt(err_e_sq), np.sqrt(err_p_sq))

    def calculate_exact(
        self, state: Statevector | QuantumCircuit
    ) -> tuple[float, float]:
        n_e, n_e_err = self._exactOperatorExpectation(state, self.n_e_obs)
        n_p, n_p_err = self._exactOperatorExpectation(state, self.n_p_obs)
        return (n_e, n_p), (n_e_err, n_p_err)


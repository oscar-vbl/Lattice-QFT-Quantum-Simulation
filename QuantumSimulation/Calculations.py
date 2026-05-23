"""
Module with functions to calculate observables,
as well as generic operations, such as expectation values, fidelity and amplitude between states.
"""

from Utils import getTimer
from Operators import (
    gauss_operator,
    buildChargeOperatorMinimal,
    buildPairCreationOperators,
    buildElectricFieldOperators,
)
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.quantumcircuit import QuantumCircuit, ClassicalRegister
from qiskit.primitives import (
    BaseEstimatorV2,
    BaseSamplerV2,
    PubResult,
    SamplerPubResult,
)
import numpy as np

from abc import ABC, abstractmethod
from typing import Iterable


class BaseObservable(ABC):
    """
    Base class for observables.

    Each calculation returns a tuple (value, error), where value is the calculated value of the observable and error is an estimation of the error of the calculation.

    Abstract methods
    ----------------
    - **get_pub**
        Returns the tuple to send to the Estimator or Sampler for the calculation of the observable.
    - **get_operators**
        Returns the list of operators that need to be measured to calculate the observable, giving them as SparsePauliOp objects.
    - **process_result_data**
        Processes the raw expectation values and standard deviations returned by the Estimator to extract the value of the observable and its error estimation.
    - **process_pub_result**
        Processes the result returned by the Estimator or Sampler to extract the value of the observable.
    - **calculate_exact**
        Calculates the value of the observable using exact methods (Statevector or expm_multiply) for ideal execution.

    Reserved methods
    ----------------
    - *Operator expectation methods:*
        * **_exactOperatorExpectation**
            Calculates the expectation value of an operator for a given state using exact methods.
        * **_pubOperatorExpectation**
            Calculates the expectation value of an operator for a given state using the result of an Estimator PUB.
        * **_pubOperatorExpectationMultiple**
            Calculates the expectation values of multiple operators for a given state using the result of an Estimator PUB.
    - *Amplitude calculation methods:*
        * **_exactAmplitudeCalculation**
            Calculates the amplitude <target_state | state> using exact methods.
        * **_getAmplitudePub**
            Returns the circuit to calculate the amplitude <target_state | state> using a Hadamard test for hardware execution.
        * **_pubAmplitudeCalculation**
            Calculates the amplitude <target_state | state> using the result of a Sampler PUB.
    - *Fidelity calculation methods:*
        * **_exactFidelityCalculation**
            Calculates the fidelity between two states using exact methods.
        * **_getFidelityPub**
            Returns the circuit to calculate the fidelity between two states using a Compute-Uncompute method for hardware execution.
        * **_pubFidelityCalculation**
            Calculates the fidelity between two states using the result of a Sampler PUB.
    """

    def __init__(self, name: str, primitive_type: str):
        self.name = name
        self.primitive_type = primitive_type  # "estimator" or "sampler"

    # pub for hardware execution (Qiskit Primitives)
    @abstractmethod
    def get_pub(self, circuit: QuantumCircuit) -> tuple:
        """
        Get the pub tuple to send to the Estimator or Sampler for the calculation of the observable.

        Returns
        -------
        tuple
            Tuple ready to send to the Estimator or Sampler for the calculation of the observable.
        """
        pass

    @abstractmethod
    def get_operators(self) -> list[SparsePauliOp]:
        """
        Get the list of operators that need to be measured to calculate the observable, giving them as SparsePauliOp objects.

        Returns
        -------
        list
            List of SparsePauliOp objects that need to be measured to calculate the observable.
        """
        pass

    @abstractmethod
    def process_result_data(self, evs: float | Iterable, stds: float | Iterable) -> tuple:
        """
        Processes the raw expectation values and standard deviations returned by the Estimator to extract the value of the observable and its error estimation.

        Parameters
        ----------
        evs : float or Iterable
            Value or list of values of expectation values returned by the Estimator for the operators needed to calculate the observable.
        stds : float or Iterable
            Value or list of values of standard deviations returned by the Estimator for the operators needed to calculate the observable.

        Returns
        -------
        tuple
            Tuple of float or Iterable with the value of the observable and its error estimation.
        """
        pass

    # Process the result returned by the Estimator or Sampler to extract the value of the observable.
    @abstractmethod
    def process_pub_result(self, result: PubResult) -> tuple:
        """
        Processes the result returned by the Estimator or Sampler to extract the value of the observable.

        Returns
        -------
        tuple
            Tuple of float or Iterable with the value of the observable and its error estimation.
        """
        pass

    # For ideal execution (expm_multiply or Statevector)
    @abstractmethod
    def calculate_exact(self, statevector: Statevector | QuantumCircuit) -> float:
        """
        Calculates the exact expectation value of the observable given the state as a Statevector or QuantumCircuit.
        As it is exact, the error returned will be 0.0.

        Returns
        -------
        tuple
            Tuple of float with the value of the observable and its error estimation (0.0).

        """
        pass

    # Operator expectation methods
    def _exactOperatorExpectation(
        self, state: Statevector | QuantumCircuit, operator: SparsePauliOp
    ) -> tuple[float, float]:
        """
        Calculates the exact expectation value of the operator given the state as a Statevector or QuantumCircuit.

        Parameters
        ----------
        state: Statevector or QuantumCircuit
            State for which we want to calculate the expectation value.
        operator: SparsePauliOp
            Operator for which we want to calculate the expectation value.

        Returns
        -------
        tuple
            Tuple of float with the value of the expectation value and its error estimation (0.0).
        """
        if isinstance(state, QuantumCircuit):
            state = Statevector(state)
        return float(state.expectation_value(operator).real), 0.0

    def _pubOperatorExpectation(self, result: PubResult) -> tuple[float, float]:
        """
        Processes the result returned by the Estimator or Sampler to extract the expectation value of a single operator.

        Parameters
        ----------
        result: PubResult
            Result returned by the Estimator or Sampler.

        Returns
        -------
        tuple
            Tuple of float with the value of the expectation value and its error estimation.
        """
        return float(np.squeeze(result.data.evs).real), float(
            np.squeeze(result.data.stds).real
        )

    def _pubOperatorExpectationMultiple(
        self, result: PubResult
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Processes the result returned by the Estimator or Sampler to extract the expectation values of multiple operators.

        Parameters
        ----------
        result: PubResult
            Result returned by the Estimator or Sampler.

        Returns
        -------
        tuple
            Tuple of np.ndarray with the values of the expectation values and their error estimations.
        """
        return np.squeeze(result.data.evs).real, np.squeeze(result.data.stds).real

    # Amplitude calculation methods
    def _exactAmplitudeCalculation(
        self,
        state: Statevector | QuantumCircuit,
        target_state: Statevector | QuantumCircuit,
    ) -> tuple[float, float]:
        """
        Calculates the amplitude <target_state | state> using exact methods, where state and target_state can be given as Statevector or QuantumCircuit.

        Parameters
        ----------
        state: Statevector or QuantumCircuit
            State for which we want to calculate the amplitude.
        target_state: Statevector or QuantumCircuit
            Target state with which we want to calculate the amplitude.

        Returns
        -------
        tuple
            Tuple with the real part of the amplitude and its error estimation (0.0).
        """
        if isinstance(state, QuantumCircuit):
            state = Statevector(state)
        if isinstance(target_state, QuantumCircuit):
            target_state = Statevector(target_state)

        return target_state.inner(state), 0.0

    def _getAmplitudePub(
        self, state: QuantumCircuit, target_state: QuantumCircuit
    ) -> tuple[QuantumCircuit, QuantumCircuit]:
        """
        Returns the circuits to calculate the real and imaginary parts of the amplitude <target_state | state> using a Hadamard test for hardware execution.

        Parameters
        ----------
        state: QuantumCircuit
            State for which we want to calculate the amplitude.
        target_state: QuantumCircuit
            Target state with which we want to calculate the amplitude.

        Returns
        -------
        tuple
            Tuple with the circuits to calculate the real and imaginary parts of the amplitude using a Hadamard test for hardware execution.
        """
        # Hardware method: Hadamard test
        assert isinstance(state, QuantumCircuit), "State must be a QuantumCircuit."
        assert isinstance(target_state, QuantumCircuit), (
            "Target state must be a QuantumCircuit."
        )

        n_qubits = state.num_qubits

        # System is in state  |0 ... 0>.
        # 1. We create operator W = U_target^-1 * U_state
        # We want to measure <0| W |0> = <0| U_target^-1 * U_state |0> = <target|state>
        # with <0| W |0> = a + bi = z
        W = QuantumCircuit(n_qubits)
        W.compose(state, inplace=True)
        W.compose(target_state.inverse(), inplace=True)

        # 2. Convert W in a controlled operation
        cW = W.to_gate(label="cW").control(1)

        # 3. Create 1 qubit classical register to measure ancilla
        # We use ancilla at the begining and apply Hadamard such that
        # |state> = |0> \otimes |0 ... 0> ->
        # |state> = \frac{1}{\sqrt{2}} (|0> + |1>) \otimes |0 ... 0>
        cr = ClassicalRegister(1, "meas")

        # --- CIRCUIT FOR REAL PART ---
        # Qubit 0 will be ancilla, Qubits 1 a N will be logical system
        qc_real = QuantumCircuit(n_qubits + 1)
        # |state> = |0> \otimes |0 ... 0>
        qc_real.add_register(cr)
        # H|state> = H|0> \otimes |0 ... 0> = \frac{1}{\sqrt{2}} (|0> + |1>) \otimes |0 ... 0>
        qc_real.h(0)
        # Apply controlled-W gate, with ancilla as control and system as target
        # (C-W) H |state> = \frac{1}{\sqrt{2}} (|0> \otimes |0 ... 0> + |1> \otimes W |0 ... 0>)
        qc_real.append(cW, range(n_qubits + 1))
        # Apply another Hadamard on ancilla, since H \otimes H = 1
        qc_real.h(0)
        # H (C-W) H |state> = \frac{1}{\sqrt{2}} (H|0> \otimes |0 ... 0> + H|1> \otimes W |0 ... 0>)
        # H (C-W) H |state> = \frac{1}{2} [ |0> \otimes ((1 + W) |0 ... 0>) + |1> \otimes ((1 - W) |0 ... 0>) ]
        # We measure P(0_A) = \frac{1}{4} | (1 + W) |0 ... 0> |^2
        # = \frac{1}{4} (<s|s> + <s|W|s> + <s|W^+|s> + <s|s>) = 1/4 (1 + z + z* + 1) = 1/2 (1 + Re(z)) =  1/2 (1 + a)
        # So, the real part of the measurement will be
        # a = 2 P(0_A) - 1
        qc_real.measure(0, 0)

        # --- CIRCUIT FOR IMAGINARY ---
        # It's analogous to the real part, except we insert an S^\dagger
        # so the imaginary part is extracted from the measurement (and not the real)
        qc_imag = QuantumCircuit(n_qubits + 1)
        qc_imag.add_register(cr)
        qc_imag.h(0)
        qc_imag.sdg(0)  # Extra phase -pi/2 changes measurement basis
        qc_imag.append(cW, range(n_qubits + 1))
        qc_imag.h(0)
        qc_imag.measure(0, 0)

        return qc_real, qc_imag

    def _pubAmplitudeCalculation(
        self, result: SamplerPubResult
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Processes the result returned by the Sampler to extract the real and imaginary parts of the amplitude <target_state | state> and their error estimations.

        Parameters
        ----------
        result: SamplerPubResult
            Result returned by the Sampler.

        Returns
        -------
        tuple
            Tuple with the real and imaginary parts of the amplitude and their error estimations, as (real_part, imag_part), (real_error, imag_error).
        """
        # Detailed process on calculateAmplitude function
        assert len(result) == 2, (
            "Result must contain two entries: one for the real part and one for the imaginary part of the amplitude."
        )
        # Extract probabilities to measure 0 on ancilla
        # Real part
        counts_real = result[0].data.meas.get_counts()
        shots_real = sum(counts_real.values())
        p0_real = counts_real.get("0", 0) / shots_real
        real_part = 2 * p0_real - 1

        # Error Real (binomial)
        shots_real = sum(counts_real.values())
        err_p0_real = (
            np.sqrt(p0_real * (1 - p0_real) / shots_real) if shots_real > 0 else 0.0
        )
        err_real = 2 * err_p0_real

        # Imaginary part
        counts_imag = result[1].data.meas.get_counts()
        shots_imag = sum(counts_imag.values())
        p0_imag = counts_imag.get("0", 0) / shots_imag
        imag_part = 2 * p0_imag - 1

        # Error Imaginary (binomial)
        shots_imag = sum(counts_imag.values())
        err_p0_imag = (
            np.sqrt(p0_imag * (1 - p0_imag) / shots_imag) if shots_imag > 0 else 0.0
        )
        err_imag = 2 * err_p0_imag

        return (real_part, imag_part), (err_real, err_imag)

    # Fidelity calculation methods
    def _exactFidelityCalculation(
        self,
        state_1: Statevector | QuantumCircuit,
        state_2: Statevector | QuantumCircuit,
    ) -> float:
        """
        Calculates the fidelity between two states using exact methods, where state_1 and state_2 can be given as Statevector or QuantumCircuit.

        Parameters
        ----------
        state_1: Statevector or QuantumCircuit
            First state for which we want to calculate the fidelity.
        state_2: Statevector or QuantumCircuit
            Second state for which we want to calculate the fidelity.

        Returns
        -------
        tuple
            Tuple with the value of the fidelity and its error estimation (0.0).
        """
        if isinstance(state_1, QuantumCircuit):
            state_1 = Statevector(state_1)
        if isinstance(state_2, QuantumCircuit):
            state_2 = Statevector(state_2)
        return float(np.abs(state_1.inner(state_2)) ** 2), 0.0

    def _getFidelityPub(
        self, state_1: QuantumCircuit, state_2: QuantumCircuit
    ) -> QuantumCircuit:
        """
        Returns the circuit to calculate the fidelity between two states using a Compute-Uncompute method for hardware execution.

        Parameters
        ----------
        state_1: QuantumCircuit
            First state for which we want to calculate the fidelity.
        state_2: QuantumCircuit
            Second state for which we want to calculate the fidelity.

        Returns
        -------
        QuantumCircuit
            Circuit to calculate the fidelity between the two states using a Compute-Uncompute method for hardware execution.
        """
        # Hardware method (Compute-Uncompute)
        # state contains preparation ansatz + Trotter evolution
        assert isinstance(state_1, QuantumCircuit), (
            "State 1 must be a QuantumCircuit when using sampler."
        )
        assert isinstance(state_2, QuantumCircuit), (
            "State 2 must be a QuantumCircuit when using sampler."
        )
        # New circuit for measurement
        measure_circuit = state_1.copy()

        # Undo initial state preparation (we apply ansatz inverse)
        # initial_state_circuit is the circuit that prepared vacuum
        measure_circuit.compose(state_2.inverse(), inplace=True)

        # Measure all qubits at the end
        measure_circuit.measure_all()

        return measure_circuit

    def _pubFidelityCalculation(self, result: SamplerPubResult) -> tuple[float, float]:
        """
        Processes the result returned by the Sampler to extract the fidelity between the two states and its error estimation.

        Parameters
        ----------
        state_1: Statevector or QuantumCircuit
            First state for which we want to calculate the fidelity.
        state_2: Statevector or QuantumCircuit
            Second state for which we want to calculate the fidelity.

        Returns
        -------
        tuple
            Tuple with the value of the fidelity and its error estimation.
        """
        # Detailed process on calculateFidelity function
        # Counts (shots) of the measurement results at the end
        counts = result.data.meas.get_counts()

        # Fidelity is the probability of measuring the all-zeros state
        # which would mean both states are the same.
        # So we take the count of the all-zeros state and divide by total shots.
        # If both states are the same (supposing vacuum), we would get
        # |00...0> -> Circuit|00...0> -> Circuit^-1 Circuit|00...0> -> |00...0>
        # so we would expect to measure all-zeros with probability 1, which means both states are the same, as expected.
        total_shots = sum(counts.values())
        zeros_state = "0" * len(list(counts.keys())[0])

        fidelity = counts.get(zeros_state, 0) / total_shots

        # Error estimation (binomial distribution)
        error = (
            np.sqrt((fidelity * (1 - fidelity)) / total_shots)
            if total_shots > 0
            else 0.0
        )

        return float(fidelity), float(error)


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

    def process_result_data(self, evs, stds) -> tuple[float, float]:
        return float(evs), float(stds)

    def calculate_exact(self, state: Statevector | QuantumCircuit) -> float:
        return self._exactOperatorExpectation(state, self.hamiltonian)


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
        evs_real = np.squeeze(evs).real
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


def checkChargeSymmetry(H, e0=0):
    """
    Check if hamiltonian respects charge symmetry.

    Parameters:
    - H: Hamiltonian as a SparsePauliOp
    - e0: background electric field (default 0)

    Returns:
    - True if H respects charge symmetry, False otherwise
    - Charge operator as a SparsePauliOp
    """
    # Length of the chain
    L = len(H.input_dims())

    # Q_n = e0 + 1/2 * sum_{m=0..n} (σ^z_m + (-1)^m)
    # Q_op = Sum(q_n)
    Q_op = buildChargeOperatorMinimal(L)

    # Conmuting check: [H, Q] = H*Q - Q*H
    commutator = (H.dot(Q_op) - Q_op.dot(H)).simplify()

    coeffs = np.asarray(commutator.coeffs).ravel()
    norm_commutator = np.sqrt((2**L) * np.sum(np.abs(coeffs) ** 2))

    print(f"{getTimer()} INFO: Norm of the conmutator: [H, Q] = {norm_commutator:.2e}")

    # If norm of the conmutator is close to zero, it commutes.
    if norm_commutator < 1e-9:
        print(f"{getTimer()} INFO: Hamiltonian respects charge symmetry.")
        return True, Q_op
    else:
        print(f"{getTimer()} WARNING: Hamiltonian does not respect charge symmetry.")
        return False, Q_op


def calculateOperatorExpectation(
    state: Statevector | QuantumCircuit,
    operator: SparsePauliOp,
    estimator: BaseEstimatorV2 | None = None,
    precision: float | None = None,
) -> float:
    """
    [DEPRECATED] Use BaseObservable methods instead.

    Calculate the expectation value of an operator for a given state.

    """
    if estimator is None:
        if isinstance(state, QuantumCircuit):
            state = Statevector(state)
        return float(state.expectation_value(operator).real)
    else:
        assert isinstance(state, QuantumCircuit), (
            "Estimator provided but state is already a Statevector. If estimator is provided, state must be a QuantumCircuit."
        )
        if precision is not None:
            pub = [(state, operator, [], precision)]
        else:
            pub = [(state, operator)]
        job = estimator.run(pub)
        result = job.result()[0]
        return float(np.squeeze(result.data.evs).real)


def calculateAmplitude(
    state: QuantumCircuit | Statevector,
    target_state: QuantumCircuit | Statevector,
    sampler: BaseSamplerV2 | None = None,
) -> complex:
    """
    [DEPRECATED] Use BaseObservable methods instead.

    Calculate probability amplitude <target_state | state> as complex number.

    If Sampler is given, the interferometry Hadamard test is applied,
    using one ancilla qubit to extract amplitude's real and imaginary part.
    """
    if sampler is None:
        if isinstance(state, QuantumCircuit):
            state = Statevector(state)
        if isinstance(target_state, QuantumCircuit):
            target_state = Statevector(target_state)

        return target_state.inner(state)

    else:
        # Hardware method: Hadamard test
        assert isinstance(state, QuantumCircuit), "State must be a QuantumCircuit."
        assert isinstance(target_state, QuantumCircuit), (
            "Target state must be a QuantumCircuit."
        )

        n_qubits = state.num_qubits

        # System is in state  |0 ... 0>.
        # 1. We create operator W = U_target^-1 * U_state
        # We want to measure <0| W |0> = <0| U_target^-1 * U_state |0> = <target|state>
        # with <0| W |0> = a + bi = z
        W = QuantumCircuit(n_qubits)
        W.compose(state, inplace=True)
        W.compose(target_state.inverse(), inplace=True)

        # 2. Convert W in a controlled operation
        cW = W.to_gate(label="cW").control(1)

        # 3. Create 1 qubit classical register to measure ancilla
        # We use ancilla at the begining and apply Hadamard such that
        # |state> = |0> \otimes |0 ... 0> ->
        # |state> = \frac{1}{\sqrt{2}} (|0> + |1>) \otimes |0 ... 0>
        cr = ClassicalRegister(1, "meas")

        # --- CIRCUIT FOR REAL PART ---
        # Qubit 0 will be ancilla, Qubits 1 a N will be logical system
        qc_real = QuantumCircuit(n_qubits + 1)
        # |state> = |0> \otimes |0 ... 0>
        qc_real.add_register(cr)
        # H|state> = H|0> \otimes |0 ... 0> = \frac{1}{\sqrt{2}} (|0> + |1>) \otimes |0 ... 0>
        qc_real.h(0)
        # Apply controlled-W gate, with ancilla as control and system as target
        # (C-W) H |state> = \frac{1}{\sqrt{2}} (|0> \otimes |0 ... 0> + |1> \otimes W |0 ... 0>)
        qc_real.append(cW, range(n_qubits + 1))
        # Apply another Hadamard on ancilla, since H \otimes H = 1
        qc_real.h(0)
        # H (C-W) H |state> = \frac{1}{\sqrt{2}} (H|0> \otimes |0 ... 0> + H|1> \otimes W |0 ... 0>)
        # H (C-W) H |state> = \frac{1}{2} [ |0> \otimes ((1 + W) |0 ... 0>) + |1> \otimes ((1 - W) |0 ... 0>) ]
        # We measure P(0_A) = \frac{1}{4} | (1 + W) |0 ... 0> |^2
        # = \frac{1}{4} (<s|s> + <s|W|s> + <s|W^+|s> + <s|s>) = 1/4 (1 + z + z* + 1) = 1/2 (1 + Re(z)) =  1/2 (1 + a)
        # So, the real part of the measurement will be
        # a = 2 P(0_A) - 1
        qc_real.measure(0, 0)

        # --- CIRCUIT FOR IMAGINARY ---
        # It's analogous to the real part, except we insert an S^\dagger
        # so the imaginary part is extracted from the measurement (and not the real)
        qc_imag = QuantumCircuit(n_qubits + 1)
        qc_imag.add_register(cr)
        qc_imag.h(0)
        qc_imag.sdg(0)  # Extra phase -pi/2 changes measurement basis
        qc_imag.append(cW, range(n_qubits + 1))
        qc_imag.h(0)
        qc_imag.measure(0, 0)

        # 4. Send both circuit to sampler in the same job
        job = sampler.run([qc_real, qc_imag])
        result = job.result()

        # 5. Extract probabilities to measure 0 on ancilla
        # Real part
        counts_real = result[0].data.meas.get_counts()
        shots_real = sum(counts_real.values())
        p0_real = counts_real.get("0", 0) / shots_real
        real_part = 2 * p0_real - 1

        # Imaginary part
        counts_imag = result[1].data.meas.get_counts()
        shots_imag = sum(counts_imag.values())
        p0_imag = counts_imag.get("0", 0) / shots_imag
        imag_part = 2 * p0_imag - 1

        # construct complex number
        return complex(real_part, imag_part)


def calculateFidelity(
    state_1: Statevector | QuantumCircuit,
    state_2: Statevector | QuantumCircuit,
    sampler: BaseSamplerV2 | None = None,
) -> float:
    """
    [DEPRECATED] Use BaseObservable methods instead.

    Calculate fidelity of two given states.
    """
    if sampler is None:
        if isinstance(state_1, QuantumCircuit):
            state_1 = Statevector(state_1)
        if isinstance(state_2, QuantumCircuit):
            state_2 = Statevector(state_2)
        return float(np.abs(state_1.inner(state_2)) ** 2)
    else:
        # Hardware method (Compute-Uncompute)
        # state contains preparation ansatz + Trotter evolution
        assert isinstance(state_1, QuantumCircuit), (
            "State 1 must be a QuantumCircuit when using sampler."
        )
        assert isinstance(state_2, QuantumCircuit), (
            "State 2 must be a QuantumCircuit when using sampler."
        )
        # New circuit for measurement
        measure_circuit = state_1.copy()

        # Undo initial state preparation (we apply ansatz inverse)
        # initial_state_circuit is the circuit that prepared vacuum
        measure_circuit.compose(state_2.inverse(), inplace=True)

        # Measure all qubits at the end
        measure_circuit.measure_all()

        # Send to Sampler
        job = sampler.run([measure_circuit])
        result = job.result()[0]

        # Counts (shots) of the measurement results at the end
        counts = result.data.meas.get_counts()

        # Fidelity is the probability of measuring the all-zeros state
        # which would mean both states are the same.
        # So we take the count of the all-zeros state and divide by total shots.
        # If both states are the same (supposing vacuum), we would get
        # |00...0> -> Circuit|00...0> -> Circuit^-1 Circuit|00...0> -> |00...0>
        # so we would expect to measure all-zeros with probability 1, which means both states are the same, as expected.
        total_shots = sum(counts.values())
        zeros_state = "0" * measure_circuit.num_qubits

        fidelity = counts.get(zeros_state, 0) / total_shots
        return float(fidelity)


def calculateEnergy(
    state: Statevector | QuantumCircuit,
    hamiltonian: SparsePauliOp,
    estimator: BaseEstimatorV2 | None = None,
    precision: float | None = None,
) -> float:
    """
    [DEPRECATED] Use EnergyObservable instead.

    Calculate energy as the expectation value of the Hamiltonian for a given state.
    """
    return calculateOperatorExpectation(state, hamiltonian, estimator, precision)


def calculateVacuumPersistence(
    state: Statevector | QuantumCircuit,
    initial_state: Statevector | QuantumCircuit,
    sampler: BaseSamplerV2 | None = None,
) -> float:
    """
    [DEPRECATED] Use PersistenceObservable instead.

    Calculate vacuum persistence as the fidelity of a given state and the initial vacuum state.
    """
    return calculateFidelity(state, initial_state, sampler)


def calculateGaussLawViolation(
    state: Statevector | QuantumCircuit,
    qubits_num: int,
    estimator: BaseEstimatorV2 | None = None,
    precision: float | None = None,
) -> float:
    """
    [DEPRECATED] Use GaussLawViolationObservable instead.

    Check violation of Gauss' law as sum of the expectation value of the Gauss operator G_n
    of all the sites on the lattice. It should be 0 (or almost).
    """
    value = 0
    for n in range(qubits_num):
        op = gauss_operator(n, qubits_num) @ gauss_operator(n, qubits_num)
        value += np.abs(calculateOperatorExpectation(state, op, estimator, precision))

    return value


def calculatePairCreation(
    state: Statevector | QuantumCircuit,
    qubits_num: int,
    estimator: BaseEstimatorV2 | None = None,
    precision: float | None = None,
) -> tuple[float, float]:
    """
    [DEPRECATED] Use PairCreationObservable instead.

    Calculate the number of pairs created as the sum of the occupation numbers of all sites.
    The occupation number of a site is calculated as n_occ = (1 + <Z>) / 2,
    where <Z> is the expectation value of the Z operator on that site.
    For even sites (electrons) we count the number of electrons created as 1 - n_occ,
    while for odd sites (positrons) we count the number of positrons created as n_occ.
    """
    n_e_obs, n_p_obs = buildPairCreationOperators(qubits_num)
    n_e = calculateOperatorExpectation(state, n_e_obs, estimator, precision)
    n_p = calculateOperatorExpectation(state, n_p_obs, estimator, precision)
    # Number of electrons and positrons
    return n_e, n_p


def calculateElectricField(
    state: Statevector | QuantumCircuit,
    qubits_num: int,
    e0: float = 0,
    estimator: BaseEstimatorV2 | None = None,
    precision: float | None = None,
) -> np.array:
    """
    [DEPRECATED] Use ElectricFieldObservable instead.

    Calculate the electric field at each link as E(n) = E_0 + sum_{k=0..n} Q_k, where Q_k is the charge operator at site k. Returns a list of the electric field at each link.
    """
    # TODO: Implement efficiently and add estimator support
    electric_fields = measure_electric_field(state, qubits_num, e0)

    return electric_fields


def measure_electric_field(state, L, e0):
    """
    Measure the electric field at each link as E(n) = E_0 + sum_{k=0..n} Q_k, where Q_k is the charge operator at site k.

    Source: https://arxiv.org/pdf/1605.04570 (Martinez et al., 2016)
    """
    E_links = []
    cumulative = 0.0
    for n in range(L - 1):
        obs_z = SparsePauliOp.from_sparse_list([("Z", [n], 1.0)], num_qubits=L)
        sz = state.expectation_value(obs_z).real
        cumulative += 0.5 * (sz + (-1) ** n)
        E_links.append(e0 + cumulative)
    return np.array(E_links)

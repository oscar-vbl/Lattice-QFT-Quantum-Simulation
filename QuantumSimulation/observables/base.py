from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.quantumcircuit import QuantumCircuit, ClassicalRegister
from qiskit.primitives import (
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
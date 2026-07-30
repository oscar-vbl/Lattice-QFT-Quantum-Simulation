"""
Main file for Schwinger simulation.
"""
import sys
import numpy as np
import pandas as pd
import time
from typing import Any, Mapping, Iterable
from tqdm.auto import tqdm
from qiskit.circuit.quantumcircuit import QuantumCircuit
from .core.transpilation import apply_transpilation
from .circuitBuilder import buildCircuit
from .Operators import buildSchwingerHamiltonianTemporalGauge
from .Calculations import checkChargeSymmetry
from .Utils import getTimer, func_return
from .observables import (
    BaseObservable,
    EnergyObservable,
    PersistenceObservable,
    PairCreationObservable,
    GaussLawViolationObservable,
    ElectricFieldObservable,
)
from .Ansatzes import (
    build_schwinger_hva,
    build_schwinger_hva_full,
    build_schwinger_hva_balanced,
)
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import efficient_su2, n_local, excitation_preserving
from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.primitives import (
    BaseEstimatorV2,
    BaseSamplerV2,
    StatevectorEstimator,
    StatevectorSampler,
)
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply


class SchwingerSimulation:
    """
    Main class for Schwinger simulation.

    It takes a configuration dictionary as input and runs the simulation according to the parameters specified in the configuration.

    It builds a lattice hamiltonian, finds the vacuum state with a VQE-like optimization, and optionally then evolves the state in time according to the specified temporal evolution parameters, calculating observables along the way.

    Full simulation is performed by calling the run_simulation() method.

    When initialized, it only stores the configuration and prepares the backend configuration if specified.

    Simulation config must have the following entries:

    Parameters
    -------------------------------
    "QubitsNumber": int
        Number of qubits in the simulation (Lattice size).
    "Hamiltonian": dict
        Parameters for the initial hamiltonian construction.
    "Ansatz": dict
        Parameters for the ansatz construction and vacuum state optimization.
    "Temporal Evolution": optional, dict
        Parameters for the temporal evolution.
    "Backend": optional, dict
        Parameters for the quantum backend.

    Workflow
    --------
    1. Initialize with configuration.
    2. run_simulation() method performs:

       a. Build initial hamiltonian according to parameters, calling self.get_hamiltonian().

       b. Check symmetries and add charge penalty if specified.

       c. Build ansatz circuit according to parameters, calling self.get_ansatz().

       d. Optimize ansatz parameters to find vacuum state, obtaining the initial state and vacuum energy, calling self.get_vacuum().

       e. If temporal evolution is active, evolve the state according to the specified parameters, calling self.evolve_state().
    """

    def __init__(
        self,
        simulation_config: Mapping[str, Any],
        initial_state: QuantumCircuit | Statevector | None = None,
    ):

        print("\n" + "#" * 70)
        print(f"{getTimer()} INFO: Initializing SchwingerSimulation class.")
        # Configuration parameters
        self.config = simulation_config
        # Number of qubits
        self.qubits_num = self.config["QubitsNumber"]
        # Initial state (if given, it will be used instead of optimizing for the vacuum state)
        self.initial_state = initial_state
        # Get sampler and estimator based on backend configuration if provided
        self.backend, self.estimator, self.sampler = self.setup_quantum_primitives()

    def run_simulation(self):
        """
        Main method to run the simulation. It performs the following steps:

        1. Build initial hamiltonian according to parameters, calling self.get_hamiltonian().

        2. Check symmetries and add charge penalty if specified.

        3. Build ansatz circuit according to parameters, calling self.get_ansatz().

        4. Optimize ansatz parameters to find vacuum state, obtaining the initial state and vacuum energy, calling self.get_vacuum().

        5. If temporal evolution is active, evolve the state according to the specified parameters, calling self.evolve_state().

        """

        print(f"{getTimer()} INFO: Starting simulation.")
        #######################################
        # 1. Get main params for the simulation
        #######################################
        # Get hamiltonian (t < 0 for later quench)
        print(
            f"{getTimer()} INFO: Parameters of the hamiltonian: {self.config['Hamiltonian']['Parameters']}"
        )
        self.hamiltonian_prep = self.get_hamiltonian()
        if self.hamiltonian_prep is None:
            raise ValueError("Failed to construct the Hamiltonian.")

        # Check symmetries
        hasSym, Q_op = checkChargeSymmetry(self.hamiltonian_prep)
        if not hasSym:
            raise ValueError(
                "The Hamiltonian does not have charge symmetry, review the Hamiltonian construction or the symmetry checking function."
            )

        # Add charge penalty in order to minimize charge symmetry violations
        lambda_penalty = self.config["Hamiltonian"].get("Lambda_Charge_Penalty", 0)
        if lambda_penalty == "Variable":
            max_coef = self.hamiltonian_prep.coeffs.max()
            lambda_penalty = max_coef * 10
            print(
                f"{getTimer()} INFO: Lambda_Charge_Penalty set to variable value: {lambda_penalty:.2e} (10 times the max coefficient of the Hamiltonian)"
            )

        if lambda_penalty > 0:
            print(
                f"{getTimer()} INFO: Adding charge penalty term with lambda = {lambda_penalty}"
            )
            self.hamiltonian_prep = (
                self.hamiltonian_prep + lambda_penalty * Q_op.dot(Q_op)
            ).simplify()
            self.hamiltonian_prep: SparsePauliOp

        # Get ansatz from config:
        # Initial config and type
        self.ansatz = self.get_ansatz()

        # ISA Transpilation for primitives V2
        # Transpile if we use a real estimator or a simulator with backend
        if self.estimator is not None and self.backend is not None:
            print(f"{getTimer()} INFO: Transpiling ansatz to ISA circuit...")
            self.ansatz, layout_circuits = apply_transpilation(
                backend=self.backend,
                circuit=self.ansatz,
                layout_circuits=[self.hamiltonian_prep]
            )
            self.hamiltonian_prep = layout_circuits[0]

        # Get parameters that minimize initial ansatz energy
        if self.initial_state is None:
            self.initial_state, self.vacuum_energy, self.vacuum_parameters = (
                self.get_vacuum()
            )
        else:
            print(f"{getTimer()} INFO: Transpiling initial state to ISA circuit...")
            self.initial_state, layout_circuits = apply_transpilation(
                backend=self.backend,
                circuit=self.initial_state,
                layout_circuits=[self.hamiltonian_prep]
            )
            self.hamiltonian_prep = layout_circuits[0]
            energy_obs = EnergyObservable(self.hamiltonian_prep, self.precision)
            if hasattr(self, "estimator") and self.estimator is not None:
                pub = energy_obs.get_pub(self.initial_state)
                result = self.estimator.run([pub]).result()
                self.vacuum_energy = energy_obs.process_pub_result(result[0])[0]
            else:
                self.vacuum_energy = energy_obs.calculate_exact(self.initial_state)[0]

            self.vacuum_parameters = None

        print(f"{getTimer()} INFO: Initial energy = {self.vacuum_energy}")

        # Ground state diagnostics
        if self.config["Ansatz"].get("Ensure_Zero_Charge", False):
            Q_total_inicial = (
                Statevector(self.initial_state).expectation_value(Q_op).real
            )
            print(f"{getTimer()} INFO: Initial charge = {Q_total_inicial}")
            if abs(Q_total_inicial) > 0.01:
                print(
                    f"{getTimer()} WARNING: Initial state does not have zero charge, review ansatz and minimization parameters."
                )
                raise ValueError(
                    "Initial state does not have zero charge, review ansatz and minimization parameters."
                )

        # Store diagnostics for later inspection
        self.ground_state_diagnostics = {
            "vacuum_energy": self.vacuum_energy,
            "vacuum_charge": Q_total_inicial
            if self.config["Ansatz"].get("Ensure_Zero_Charge", False)
            else None,
            "optimization_history": self.optimization_history
            if hasattr(self, "optimization_history")
            else [],
        }

        #######################
        # 2. Temporal evolution
        #######################

        if self.config.get("Temporal Evolution", {}).get("Active", False):
            (
                self.final_state,
                self.evolution_data,
                self.evolution_error,
                self.all_trotter_evolution_data,
                self.all_trotter_evolution_error,
            ) = self.evolve_state()

        # Additional results and calculations (not developed in this module)
        # Check Results for results examples
        print(f"{getTimer()} INFO: Simulation ended.")
        print("#" * 70 + "\n")

    def setup_quantum_primitives(self):
        """
        Initializes and returns the Backend and the appropriate Qiskit V2 Primitives (Estimator and Sampler)
        based on the backend configuration. Supports Aer simulators and real IBM Quantum hardware.
        """
        backend_config    = self.config.get("Backend", {})
        self.backend_type = backend_config.get("Type", None)
        if not backend_config or not self.backend_type:
            self.backend_type = None
            self.precision = None
            print(
                f"{getTimer()} INFO: No backend specified. Evolving with direct matrix operations."
            )
            return None, None, None

        
        backend_options = backend_config.get("Options", {})
        self.precision  = backend_config.get("Precision", None)
        shots           = backend_config.get("Shots", None)
        noise_model     = backend_config.get("Noise Model", None)

        if self.backend_type == "StatevectorEstimator":
            print(
                f"{getTimer()} INFO: Using StatevectorEstimator (Ideal V2 Primitive)."
            )
            # Precision of the estimator is 1/sqrt(N), with N=shots
            if shots is not None:
                est_params = {"default_precision": 1 / np.sqrt(shots)}
                samp_params = {"default_shots": shots}
            else:
                est_params = {}
                samp_params = {}
            return (
                None,
                StatevectorEstimator(**est_params),
                StatevectorSampler(**samp_params),
            )

        elif self.backend_type in ["Aer", "AerSimulator"]:
            print(f"{getTimer()} INFO: Using Primitives V2 backed by AerSimulator.")
            if noise_model is not None:
                print(f"{getTimer()} INFO: Applying noise model to AerSimulator.")
                aer_noise_model = NoiseModel()
                for error_params in noise_model.get("Errors", []):
                    error_type = error_params.get("Type")
                    num_qubits = error_params.get("Num_Qubits", 1)
                    gates      = error_params.get("Gates", [])
                    if error_type == "Depolarizing":
                        error_param = depolarizing_error(
                            error_params.get("Probability", 0.01),
                            num_qubits,
                        )
                    else:
                        print(
                            f"{getTimer()} WARNING: Noise model error type {error_type} not recognized. Skipping this error."
                        )
                        continue
                    aer_noise_model.add_all_qubit_quantum_error(
                        error_param, gates
                    )
            else:
                aer_noise_model = None

            aer_backend = AerSimulator(noise_model=aer_noise_model)

            # Config estimator options
            estimator_options: dict = {}
            # Add noise model to backend options if specified
            if aer_noise_model is not None:
                estimator_options["backend_options"] = {"noise_model": aer_noise_model}

            # Add additional backend options from config if specified
            if backend_config.get("Backend Options"):
                estimator_options.setdefault("backend_options", {}).update(
                    backend_config["Backend Options"]
                )
            # Shots config
            if shots is not None:
                # Add shots if specified
                estimator_options["run_options"] = {"shots": shots}
                estimator_options["default_precision"] = 1 / np.sqrt(shots)
            elif self.precision is not None:
                # Add default precision if shots not given
                estimator_options["default_precision"] = self.precision

            estimator = EstimatorV2(options=estimator_options)

            # Config sampler options
            sampler_options = {}
            if aer_noise_model is not None:
                sampler_options["backend_options"] = {"noise_model": aer_noise_model}
            if backend_config.get("Backend Options"):
                sampler_options.setdefault("backend_options", {}).update(
                    backend_config["Backend Options"]
                )

            sampler = SamplerV2(
                default_shots=shots if shots is not None else 1024,
                options=sampler_options if sampler_options else None
            )

            return aer_backend, estimator, sampler

        elif self.backend_type == "IBM_Runtime":
            # Import inside the function to avoid dependency if not used
            from qiskit_ibm_runtime import QiskitRuntimeService
            from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2
            from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2

            print(f"{getTimer()} INFO: Connecting to IBM Qiskit Runtime Service...")
            service = QiskitRuntimeService()

            target_hardware = backend_config.get("Hardware_Name", "least_busy")
            if target_hardware == "least_busy":
                backend = service.least_busy(operational=True, simulator=False)
            else:
                backend = service.backend(target_hardware)

            print(f"{getTimer()} INFO: Target hardware selected: {backend.name}")

            estimator = RuntimeEstimatorV2(mode=backend)
            sampler = RuntimeSamplerV2(mode=backend)

            # Apply resilience levels if specified
            estimator.options.resilience_level = backend_options.get(
                "resilience_level", 1
            )

            return backend, estimator, sampler

        else:
            print(
                f"{getTimer()} WARNING: Backend type {self.backend_type} not recognized. Reverting to exact multiplication."
            )
            raise NotImplementedError(
                f"Backend type {self.backend_type} not recognized."
            )

    def get_hamiltonian(self, override_params: dict | None = None) -> SparsePauliOp:
        """
        Get hamiltonian according to the entry "Hamiltonian" in self.config.

        Parameters
        ----------
        override_params: dict, optional
            Dictionary of parameters to overwrite the ones in the configuration for the hamiltonian construction. This is useful for quenches in the temporal evolution, where the hamiltonian parameters change after t=0. The keys of the dictionary should match the ones in self.config["Hamiltonian"]["Parameters"].

        Required Attributes
        -------------------
        self.config: dict
            Configuration dictionary for the simulation, with the following relevant entries in self.config["Hamiltonian"]:
            * "Type" (str): type of hamiltonian to build (e.g. "Schwinger").
            * "Gauge" (str): gauge to use for the hamiltonian construction. Default "Temporal".
            * "Parameters" (dict): parameters for the hamiltonian construction. Lattice parameters:
                - "L" (int): lattice size (number of sites).
                - "m" (float): fermion mass.
                - "a" (float): lattice spacing.
                - "e0" (optional, float): background field.
        self.qubits_num: int
             Number of qubits in the simulation, needed for the hamiltonian construction.

        Returns
        -------
        SparsePauliOp
             The hamiltonian operator in the form of a sparse Pauli sum.
        """
        ham_type = self.config["Hamiltonian"]["Type"]
        ham_params = self.config["Hamiltonian"]["Parameters"].copy()
        ham_gauge = self.config["Hamiltonian"].get("Gauge", "Temporal")

        # Overwrite quench params if applies
        if override_params:
            ham_params.update(override_params)

        if ham_type == "Schwinger":
            if ham_gauge == "Temporal":
                return func_return(
                    buildSchwingerHamiltonianTemporalGauge,
                    ham_params,
                    expect_type=SparsePauliOp,
                )
            else:
                print(f"{getTimer()} WARNING: Gauge {ham_gauge} not implemented...")
                raise NotImplementedError(f"Gauge {ham_gauge} not implemented.")
        else:
            print(
                f"{getTimer()} WARNING: Hamiltonian type {ham_type} not implemented..."
            )
            raise NotImplementedError(f"Hamiltonian type {ham_type} not implemented.")

    def get_ansatz(self):
        """
        Build initial ansatz circuit according to the entry "Ansatz" in self.config.

        Required Attributes
        -------------------
        self.config: dict
            Configuration dictionary for the simulation, with the following relevant entries in self.config["Ansatz"]:
            - "Type": str, type of ansatz to build (e.g. "EfficientSU2", "ExcitationPreserving", "TwoLocal", "HVA", "HVA Full", "HVA Simple").
            - "Initial State": optional, dict, parameters for the initial state preparation. Initial state parameters:
                - "Vacuum": bool (default: True), whether to prepare the vacuum state as initial state.
                - "Staggered": bool (default: True), whether to use a staggered configuration for the initial state (electrons in even sites and positrons in odd sites).
            - "Entanglement": str (default: "linear"), type of entanglement for the ansatz (e.g. "linear", "full", etc.).
            - "Reps": int (default: 3), number of repetitions for the ansatz layers.
            - "AdditionalParams": dict (default: {}), additional parameters for the ansatz construction if needed.

        Returns
        -------
        QuantumCircuit
            The ansatz circuit built according to the specified parameters.
        """
        ansatz_type = self.config["Ansatz"]["Type"]
        in_state_params = self.config["Ansatz"].get("Initial State", {})

        if in_state_params.get("Vacuum", True):
            in_state_config = {
                "QubitsNumber": self.qubits_num,
            }
            if in_state_params.get("Staggered", True):
                in_state_config["Gates"] = []
                for qubit_num in range(0, self.qubits_num, 2):
                    in_state_config["Gates"] += [
                        {"gate": "X", "qubit": {"Number": qubit_num}}
                    ]
                initial_circuit = buildCircuit(in_state_config)
            else:
                print(
                    f"{getTimer()} WARNING: Ansatz state for not staggered fermions not implemented..."
                )
                raise NotImplementedError(
                    "Ansatz state for not staggered fermions not implemented."
                )
        else:
            print(
                f"{getTimer()} WARNING: Ansatz state for not vacuum state not implemented..."
            )
            raise NotImplementedError(
                "Ansatz state for not vacuum state not implemented."
            )

        if ansatz_type == "EfficientSU2":
            ansatz_func = efficient_su2
        elif ansatz_type == "ExcitationPreserving":
            ansatz_func = excitation_preserving
        elif ansatz_type == "TwoLocal":
            ansatz_func = n_local
        elif ansatz_type == "HVA":
            ansatz_func = build_schwinger_hva_balanced
            if self.config["Ansatz"].get("AdditionalParams", {}):
                self.config["Ansatz"]["AdditionalParams"] = {
                    **self.config["Ansatz"]["AdditionalParams"],
                    **{"hamiltonian": self.hamiltonian_prep},
                }
            else:
                self.config["Ansatz"]["AdditionalParams"] = {
                    "hamiltonian": self.hamiltonian_prep
                }
        elif ansatz_type == "HVA Full":
            ansatz_func = build_schwinger_hva_full
            if self.config["Ansatz"].get("AdditionalParams", {}):
                self.config["Ansatz"]["AdditionalParams"] = {
                    **self.config["Ansatz"]["AdditionalParams"],
                    **{"hamiltonian": self.hamiltonian_prep},
                }
            else:
                self.config["Ansatz"]["AdditionalParams"] = {
                    "hamiltonian": self.hamiltonian_prep
                }
        elif ansatz_type == "HVA Simple":
            ansatz_func = build_schwinger_hva
            if self.config["Ansatz"].get("AdditionalParams", {}):
                self.config["Ansatz"]["AdditionalParams"] = {
                    **self.config["Ansatz"]["AdditionalParams"],
                    **{"hamiltonian": self.hamiltonian_prep},
                }
            else:
                self.config["Ansatz"]["AdditionalParams"] = {
                    "hamiltonian": self.hamiltonian_prep
                }
        else:
            print(f"{getTimer()} WARNING: Ansatz type {ansatz_type} not implemented...")
            raise NotImplementedError(f"Ansatz type {ansatz_type} not implemented.")

        ansatz_params = {
            "num_qubits": self.qubits_num,
            "entanglement": self.config["Ansatz"].get("Entanglement", "linear"),
            "reps": self.config["Ansatz"].get("Reps", 3),
        }

        print(
            f"{getTimer()} INFO: Using ansatz {ansatz_type} with parameters: {ansatz_params}"
        )

        ansatz_params = {
            **ansatz_params,
            **self.config["Ansatz"].get("AdditionalParams", {}),
        }

        ansatz = func_return(ansatz_func, ansatz_params)
        assert isinstance(ansatz, QuantumCircuit), (
            "WARNING: Ansatz function did not return a QuantumCircuit, review ansatz parameters..."
        )
        # Evolve from initial vacuum
        full_ansatz_circuit = initial_circuit.compose(ansatz)
        full_ansatz_circuit: QuantumCircuit
        return full_ansatz_circuit

    def get_vacuum(self):
        """
        Optimize ansatz parameters to find the vacuum state, minimizing the energy cost function, according to parameters in self.config["Ansatz"].

        Some parameters on SchwingerSimulation must have been initialized before.

        Required Attributes
        ----------
        self.config: dict
            Configuration dictionary for the simulation, with the following relevant entries in self.config["Ansatz"]:
            * "Minimizer" (dict): Parameters for the minimization algorithm. Minimizer parameters:
                - "Method" (str): Optimization method to use (e.g. "COBYLA", "Nelder-Mead", "BFGS", etc.). Default: "COBYLA".
                - "Options" (optional, dict): options for the optimization method (e.g. {"maxiter": 1000, "tol": 1e-6}). Default: {"maxiter": 2000, "tol": 1e-6}.
                - "AdditionalParams" (optional, dict): additional parameters for the optimization method if needed.
            * "Initial Parameters" (optional, array-like): initial parameters for the optimization. If not given, it is initialized randomly (with a fixed seed for reproducibility). Initial parameters strategy can be configured with "Init_Strategy" (default: "random_small", which initializes parameters with small random values near zero, better for convergence in VQE-like optimizations).
            * "Init_Strategy" (optional, str): strategy for initializing parameters if "Initial Parameters" is not given. Options: "random_small", "uniform_random". Default: "random_small".
        self.ansatz: QuantumCircuit
            The ansatz circuit for the vacuum state optimization, built according to self.get_ansatz()
        self.hamiltonian_prep: SparsePauliOp
            The hamiltonian operator in the form of a sparse Pauli sum, built according to self.get_hamiltonian().

        Returns
        -------
        vacuum_state: QuantumCircuit
            The statevector of the optimized vacuum state.
        vacuum_energy: float
            The energy of the optimized vacuum state.
        vacuum_parameters: array-like
            The parameters of the ansatz that minimize the energy cost function.
        """
        # Assert needed variables are defined
        assert hasattr(self, "ansatz") and self.ansatz is not None, (
            "Ansatz not defined, cannot optimize vacuum state. Make sure to call get_ansatz() before get_vacuum() or assign it explicitly."
        )
        assert (
            hasattr(self, "hamiltonian_prep") and self.hamiltonian_prep is not None
        ), (
            "Hamiltonian not defined, cannot optimize vacuum state. Make sure to call get_hamiltonian() before get_vacuum() or assign it explicitly."
        )

        initial_state_params = self.config["Ansatz"].get("Initial Parameters", None)
        if not initial_state_params:
            # np.random.seed(42)
            # IMPROVED: Better parameter initialization (near identity instead of uniform random)
            init_strategy = self.config["Ansatz"].get("Init_Strategy", "random_small")
            init_max = self.config["Ansatz"].get("Init_Max", 0.1)
            if init_strategy == "random_small":
                # Start near identity: better convergence for VQE
                initial_state_params = np.random.normal(
                    0, init_max, self.ansatz.num_parameters
                )
            elif init_strategy == "zeros":
                # Start near identity: better convergence for VQE
                initial_state_params = np.zeros(self.ansatz.num_parameters)
            elif init_strategy == "uniform_random":
                # Original uniform random
                initial_state_params = (
                    np.random.random(self.ansatz.num_parameters) * 2 * np.pi
                )
            else:
                # Default to small random
                initial_state_params = np.random.normal(
                    0, 0.1, self.ansatz.num_parameters
                )

        if not self.config["Ansatz"].get("Minimizer", None):
            self.config["Ansatz"]["Minimizer"] = {}

        minimizer_method = self.config["Ansatz"]["Minimizer"].get("Method", "COBYLA")
        # IMPROVED: Increased default maxiter from 100 to 2000
        if minimizer_method == "L-BFGS-B":
            default_options = {"maxiter": 2000, "gtol": 1e-6}
        else:
            default_options = {"maxiter": 2000, "tol": 1e-6}
        user_options = self.config["Ansatz"]["Minimizer"].get("Options", {})
        # Merge: user options override defaults
        final_options = {**default_options, **user_options}

        minimize_params = {
            "method": minimizer_method,
            "options": final_options,
            **self.config["Ansatz"]["Minimizer"].get("AdditionalParams", {}),
        }

        if self.config["Ansatz"].get("Use Gradient Cost", False):
            if hasattr(self, "estimator"):
                self.gradient_estimator = self.estimator
            else:
                self.gradient_estimator = StatevectorEstimator()
            minimize_params["jac"] = self.gradient_cost_function
            print(f"{getTimer()} INFO: Analytic gradient (ParamShift) enabled.")

        print(
            f"{getTimer()} INFO: Minimization config: method={minimize_params['method']}, maxiter={final_options['maxiter']}"
        )
        print(f"{getTimer()} INFO: Initial parameters strategy: {init_strategy}")

        # IMPROVED: Track optimization history for diagnostics
        self.optimization_history = []

        def callback(xk):
            energy = self.energy_cost_function(xk)
            self.optimization_history.append(energy)
            # Update progress bar
            try:
                pbar.update(1)
            except:
                pass

        max_iter = final_options.get("maxiter", 2000)
        # Initiate energy observable for callback access (needs to be after estimator is defined)
        self._energy_cost = EnergyObservable(self.hamiltonian_prep, self.precision)
        vqe_start = time.time()
        # Run VQE
        # If backend is IBM Runtime, we need to open a session
        # and use the session estimator for the optimization,
        # then restore the original estimator for the rest of the simulation.
        # This is because the RuntimeEstimator needs to be used within a session context
        # to work properly, and we want to keep the session open
        # only for the duration of the VQE optimization to avoid unnecessary costs and resource usage.
        # Session is used to avoid making a single call (and wait) for every energy measurement
        if self.backend_type == "IBM_Runtime":
            # Import inside the function to avoid dependency if not used
            from qiskit_ibm_runtime import Session
            from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2

            print(
                f"{getTimer()} INFO: Opening IBM Runtime Session for VQE optimization..."
            )
            # Open exclusive VQE session
            with Session(backend=self.backend) as session:
                # New estimator for this session
                session_estimator = RuntimeEstimatorV2(mode=session)
                session_estimator.options.resilience_level = (
                    self.estimator.options.resilience_level
                )

                # We change the original estimator so the Session uses the one associated with it
                original_estimator = self.estimator
                self.estimator = session_estimator

                # Launch VQE
                with tqdm(
                    total=max_iter,
                    desc="VQE Optimization (Session)",
                    unit="iter",
                    file=sys.stdout,
                ) as pbar:
                    result = minimize(
                        self.energy_cost_function,
                        initial_state_params,
                        **minimize_params,
                        callback=callback,
                    )
                    if hasattr(result, "nit"):
                        pbar.total = pbar.n = result.nit
                        pbar.refresh()

                # Get back original estimator for the rest of the simulation
                self.estimator = original_estimator
                print(f"{getTimer()} INFO: IBM Runtime Session closed successfully.")
        else:
            with tqdm(
                total=max_iter,
                desc="VQE Optimization",
                unit="iter",
                file=sys.stdout,
                leave=True,
                dynamic_ncols=False,
            ) as pbar:
                result = minimize(
                    self.energy_cost_function,
                    initial_state_params,
                    **minimize_params,
                    callback=callback,
                )
                # Update bar to show actual iterations
                if hasattr(result, "nit"):
                    pbar.total = result.nit
                    pbar.n = result.nit
                    pbar.refresh()

        self.vqe_duration = time.time() - vqe_start
        self.vqe_iters = result.nit if hasattr(result, "nit") else None
        self.vqe_success = result.success

        vacuum_parameters = result.x
        vacuum_state = self.ansatz.assign_parameters(vacuum_parameters)
        vacuum_energy = result.fun

        # IMPROVED: Diagnostic information
        print(f"{getTimer()} INFO: Optimization completed.")
        print(f"{getTimer()} INFO:   Converged: {result.success}")
        print(
            f"{getTimer()} INFO:   Iterations: {result.nit if hasattr(result, 'nit') else 'N/A'}"
        )
        print(f"{getTimer()} INFO:   Final energy: {vacuum_energy:.8f}")
        if len(self.optimization_history) > 1:
            energy_improvement = (
                self.optimization_history[0] - self.optimization_history[-1]
            )
            print(f"{getTimer()} INFO:   Energy improvement: {energy_improvement:.8f}")

        return vacuum_state, vacuum_energy, vacuum_parameters

    def energy_cost_function(self, params: Mapping | Iterable):
        """
        Cost funcion to minimize energy of the state
        obtained by the ansatz with given parameters,
        given self.hamiltonian_prep and self.ansatz.

        Parameters
        ----------
        params: array-like
            Parameters to assign to the ansatz for the energy evaluation.

        Returns
        -------
        energy: float
            Energy of the state obtained by the ansatz with given parameters, calculated with self.hamiltonian_prep. Calculated with self.estimator if given, else calculated exactly.
        """
        ansatz_circuit = self.ansatz.assign_parameters(params)
        if hasattr(self, "estimator") and self.estimator is not None:
            pub = self._energy_cost.get_pub(ansatz_circuit)
            result = self.estimator.run([pub]).result()
            return self._energy_cost.process_pub_result(result[0])[0]
        else:
            return self._energy_cost.calculate_exact(ansatz_circuit)[0]

    def gradient_cost_function(self, params: Mapping | Iterable):
        """
        Calculate analytical gradient using Parameter Shift Rule.

        Parameters
        ----------
        params: array-like
            Parameters to assign to the ansatz for the gradient evaluation.

        Returns
        -------
        gradient: array-like
            Gradient of the energy cost function with respect to the ansatz parameters, calculated with self.gradient_estimator.
        """
        # Gradient calculator initialization
        gradient = ParamShiftEstimatorGradient(estimator=self.gradient_estimator)

        # Run job
        job = gradient.run([self.ansatz], [self.hamiltonian_prep], [params])

        return np.array(job.result().gradients[0])

    def evolve_state(self):
        """
        Evolves the initial state with the configuration "Temporal Evolution" at self.config.

        Some parameters must have been initialized previously.

        Required Attributes
        -------------------
        self.config : dict
            Main configuration dictionary of the simulation. Must have a "Temporal Evolution" entry, with the following entries:
            * "Active" (bool): Whether to perform temporal evolution or not.
            * "Time_Steps" (int): Number of time steps for the evolution.
            * "Total_Time" (float): Total time for the evolution.
            * "Evolution_Gate" (dict): Parameters for the evolution gate construction:
                - "Type" (str): Type of evolution gate to use (e.g., "Pauli"). Default: "Pauli".
                - "Synthesis" (str): Method for the evolution gate synthesis. Default: "TrotterSuzuki".
                - "Synthesis_Params" (dict): Additional parameters for the synthesis.
            * "Quench" (dict): Parameters for a possible quench in the evolution:
                - "Active" (bool): Whether to perform a quench or not.
                - "Parameters_to_Change" (dict): Parameters to change in the hamiltonian
                  (e.g., {"m": 0.5} to change the fermion mass).
            * "Observables" (dict): Parameters for the observables to calculate:
                - "Observables_List" (list of str): Observables to calculate (e.g., ["Energy"]).
                - "Observables_Params" (dict): Additional parameters for the observables.
        self.initial_state: QuantumCircuit
            The initial state of the system, obtained from the get_vacuum() method or assigned
        self.hamiltonian_prep: SparsePauliOp
            The Hamiltonian of the system, obtained from the get_hamiltonian() method, used for the evolution.
            If a quench is applied in the evolution, the Hamiltonian used for the evolution (self.hamiltonian_evol) will be different.

        Returns
        -------
        state: QuantumCircuit or Statevector
            State at the end of the evolution. Statevector is returned only for ideal simulation with MatrixExponential evolution.

        observables_dataframe: DataFrame
            DataFrame with time values of the evolution as index
            and columns the observables given in the list in the configuration for the best Trotter configuration.

        observables_dataframe_error: DataFrame
            DataFrame with statistical error estimates of observables measurement,
            with time values of the evolution as index and columns the observables
            given in the list in the configuration for the best Trotter configuration.

        all_trotter_evolution_data: dict
            dict with the evolution data for each Trotter configuration
            if "Trotter_Steps" is given in the configuration

        all_trotter_evolution_error: dict
            dict with the statistical error estimates of the evolution data
            for each Trotter configuration if "Trotter_Steps" is given in the configuration
        """

        # Assert needed variables are defined
        assert hasattr(self, "initial_state") and self.initial_state is not None, (
            "Initial state not defined, cannot evolve state. Make sure to call get_vacuum() before evolve_state() or assign it explicitly."
        )
        assert (
            hasattr(self, "hamiltonian_prep") and self.hamiltonian_prep is not None
        ), (
            "Hamiltonian not defined, cannot optimize vacuum state. Make sure to call get_hamiltonian() before get_vacuum() or assign it explicitly."
        )

        evolution_params = self.config["Temporal Evolution"]

        time_steps = evolution_params["Time_Steps"]
        total_time = evolution_params["Total_Time"]
        step = total_time / time_steps

        evolution_gate_params = evolution_params["Evolution_Gate"]
        evolution_gate_type = evolution_gate_params.get("Type", "Pauli")
        if evolution_gate_type == "Pauli":
            gate = PauliEvolutionGate
        else:
            print(
                f"{getTimer()} WARNING: Evolution gate type {evolution_gate_type} not implemented..."
            )
            raise NotImplementedError(
                f"Evolution gate type {evolution_gate_type} not implemented."
            )
        synthesis_method = evolution_gate_params.get("Synthesis", "TrotterSuzuki")
        synthesis_params = evolution_gate_params.get("Synthesis_Params", {})
        if synthesis_method == "TrotterSuzuki":
            synthesis = SuzukiTrotter
        else:
            print(
                f"{getTimer()} WARNING: Synthesis method {synthesis_method} not implemented..."
            )
            raise NotImplementedError(
                f"Synthesis method {synthesis_method} not implemented."
            )

        print(
            f"{getTimer()} INFO: Evolution gate type {evolution_gate_type} with synthesis method {synthesis_method} selected. Synthesis parameters: {synthesis_params}"
        )

        quench_config = evolution_params.get("Quench", {})
        if quench_config.get("Active", False):
            quench_params = quench_config.get("Parameters_to_Change", {})
            print(
                f"{getTimer()} INFO: Applying Quench with parameters: {quench_params}"
            )
            self.hamiltonian_evol = self.get_hamiltonian(override_params=quench_params)
            self.e0 = quench_params.get(
                "e0", self.config["Hamiltonian"]["Parameters"].get("e0", 0)
            )
        else:
            print(
                f"{getTimer()} INFO: No Quench parameters, evolving with base hamiltonian."
            )
            self.hamiltonian_evol = self.hamiltonian_prep
            self.e0 = self.config["Hamiltonian"]["Parameters"].get("e0", 0)

        # Define initial state
        if self.estimator is None:
            state = Statevector.from_instruction(self.initial_state)
        else:
            state = self.initial_state.copy()
            estimator_pubs = {"pubs": [], "info": [], "lengths": [], "trotter_key": []}
            sampler_pubs = {"pubs": [], "info": [], "lengths": [], "trotter_key": []}

        # Get Trotter configurations if defined for different time steps
        trotter_configs = evolution_params.get("Trotter_Steps", None)
        if not trotter_configs:
            trotter_configs = {"dt": {"step_multiplier": 1.0}}
        trotter_circuits = {}
        states = {}

        for key, t_conf in trotter_configs.items():
            multiplier = t_conf.get("step_multiplier", 1.0)
            current_time_step = step * multiplier
            evolution_gate = gate(
                self.hamiltonian_evol,
                time=current_time_step,
                synthesis=synthesis(**synthesis_params),
            )

            # Construct Trotter circuit to compose
            trotter_step_circuit = QuantumCircuit(self.qubits_num)
            trotter_step_circuit.append(evolution_gate, range(self.qubits_num))
            if self.estimator is not None:
                if self.backend is not None and self.backend_type not in [
                    "StatevectorEstimator",
                    None,
                ]:
                    print(f"{getTimer()} INFO: Transpiling Trotter step {key} to ISA...")

                    # Force PassManager to use same physical layout than VQE
                    initial_layout = (
                        self.initial_state.layout
                        if hasattr(self.initial_state, "layout")
                        else None
                    )
                    trotter_circuits[key], _ = apply_transpilation(
                        self.backend,
                        trotter_step_circuit,
                        initial_layout=initial_layout
                    )
                else:
                    # Ideal simulators
                    # Use twice decompose to ensure we get down to basic gates for the estimator/sampler
                    # PauliEvolutionGate has 2 levels of abstraction
                    # We first decompose into single exponential gates, then decompose those into basic gates
                    trotter_circuits[key] = trotter_step_circuit.decompose().decompose()
            else:
                # Ideal evolution without primitives
                # Use twice decompose to ensure we get down to basic gates
                # PauliEvolutionGate has 2 levels of abstraction
                # We first decompose into single exponential gates, then decompose those into basic gates
                trotter_circuits[key] = trotter_step_circuit.decompose().decompose()

            # Initialize states for each Trotter configuration
            states[key] = state.copy()

        # Define evolution method for null backend
        evolution_method = evolution_params.get("Evolution_Method", None)

        # Prepare evolution data structures according to backend
        if evolution_method == "MatrixExponential":
            sparse_ham = self.hamiltonian_evol.to_matrix(sparse=True)

        # Observables config dict
        observables = evolution_params.get("Observables", {})
        # List of all observables
        observables_list = observables.get("Observables_List", [])
        # Dict with spec parameters for observables {"obs": {"param1": value1, ...}, ...}
        observables_params = observables.get("Observables_Params", {})
        # Initialization of observable object of type BaseObservable
        observables_objects = self._initiate_observables(
            observables_list, observables_params
        )
        # Data structures to store observables data during evolution
        observables_raw_data = {
            key: {obs.name: [] for obs in observables_objects}
            for key in trotter_configs.keys()
        }
        observables_raw_error = {
            key: {obs.name: [] for obs in observables_objects}
            for key in trotter_configs.keys()
        }
        # Iterate over time steps
        with tqdm(
            range(time_steps + 1),
            desc="Evolving state",
            unit="step",
            file=sys.stdout,
            leave=True,
            dynamic_ncols=False,
        ) as pbar:
            for t in pbar:  # Time iteration
                for trotter_key, state in list(states.items()):
                    # Calculate observables for current state
                    step_operators = []
                    step_obs_objects = []
                    step_obs_lengths = []
                    for obs in observables_objects:
                        if self.estimator is None:
                            value, error = obs.calculate_exact(state)
                            observables_raw_data[trotter_key][obs.name].append(value)
                            observables_raw_error[trotter_key][obs.name].append(error)
                        else:
                            pub = obs.get_pub(state)
                            if obs.primitive_type == "estimator":
                                operators = obs.get_operators()
                                step_operators += operators
                                step_obs_objects += [obs]
                                step_obs_lengths += [len(operators)]
                                # estimator_pubs["pubs"]        += [pub]
                                # estimator_pubs["info"]        += [obs]
                                # estimator_pubs["trotter_key"] += [trotter_key]
                            elif obs.primitive_type == "sampler":
                                sampler_pubs["pubs"] += [pub]
                                sampler_pubs["info"] += [obs]
                                sampler_pubs["trotter_key"] += [trotter_key]

                    # Add grouped PUBs for estimator if we have more than one operator to optimize calls
                    # One Estimator PUB per Trotter step with all operators for all observables at that step
                    if step_operators:
                        grouped_pub = (state, step_operators, None, self.precision)
                        estimator_pubs["pubs"].append(grouped_pub)
                        estimator_pubs["info"].append(step_obs_objects)
                        # Len of each object, so we can unpack results later
                        # and know which value/error corresponds to which observable object
                        estimator_pubs["lengths"].append(step_obs_lengths)
                        estimator_pubs["trotter_key"].append(trotter_key)

                    # Evolve state
                    if t < time_steps:  # No need to evolve at the last step
                        multiplier = trotter_configs[trotter_key].get(
                            "step_multiplier", 1.0
                        )
                        num_applications = int(1 / multiplier)
                        trotter_step = step * multiplier
                        for _ in range(num_applications):
                            if self.estimator is None:
                                # Evolve state directly (no Aer backend)
                                if evolution_method == "MatrixExponential":
                                    state_data = expm_multiply(
                                        -1j * sparse_ham * trotter_step, state.data
                                    )
                                    state = Statevector(state_data)
                                else:
                                    # Default: use gate evolution (slower but exact)
                                    state = state.evolve(trotter_circuits[trotter_key])
                            else:
                                state.compose(
                                    trotter_circuits[trotter_key],
                                    range(self.qubits_num),
                                    inplace=True,
                                )

                        # Set evolved state for next Trotter step
                        states[trotter_key] = state.copy()

        # Process PUBS results if using estimator
        if self.estimator is not None and estimator_pubs["pubs"]:
            print(
                f"{getTimer()} INFO: Running {len(estimator_pubs['pubs'])} grouped Estimator PUBS..."
            )
            evolve_estimator_start = time.time()
            res_est = self.estimator.run(estimator_pubs["pubs"]).result()
            self.estimator_duration = time.time() - evolve_estimator_start
            print(
                f"{getTimer()} INFO: Finished running Estimator PUBS, {round(len(estimator_pubs['pubs']) / self.estimator_duration, 4)} PUBS per second."
            )

            for grouped_result, obs_list, lengths, trotter_key in zip(
                res_est,
                estimator_pubs["info"],
                estimator_pubs["lengths"],
                estimator_pubs["trotter_key"],
            ):
                # Extract qiskit arrays with all values and errors for this grouped PUB
                evs_all = np.atleast_1d(grouped_result.data.evs)
                stds_all = np.atleast_1d(grouped_result.data.stds)

                current_idx = 0
                for obs_obj, length in zip(obs_list, lengths):
                    # Take the slice of values and errors corresponding to this observable object
                    evs_slice = evs_all[current_idx : current_idx + length]
                    stds_slice = stds_all[current_idx : current_idx + length]

                    # Call processing
                    val, err = obs_obj.process_result_data(evs_slice, stds_slice)

                    observables_raw_data[trotter_key][obs_obj.name].append(val)
                    observables_raw_error[trotter_key][obs_obj.name].append(err)

                    # Take next observable
                    current_idx += length

        # Process PUBS results if using sampler
        if self.sampler is not None and sampler_pubs["pubs"]:
            print(
                f"{getTimer()} INFO: Running {len(sampler_pubs['pubs'])} Sampler PUBS..."
            )
            evolve_sampler_start = time.time()
            res_samp = self.sampler.run(sampler_pubs["pubs"]).result()
            self.sampler_duration = time.time() - evolve_sampler_start
            print(
                f"{getTimer()} INFO: Finished running Sampler PUBS, {round(len(sampler_pubs['pubs']) / self.sampler_duration, 4)} PUBS per second."
            )

            for result, obs_obj, trotter_key in zip(
                res_samp, sampler_pubs["info"], sampler_pubs["trotter_key"]
            ):
                value, error = obs_obj.process_pub_result(result)
                observables_raw_data[trotter_key][obs_obj.name].append(value)
                observables_raw_error[trotter_key][obs_obj.name].append(error)

        # Dict to store DataFrames for each Trotter configuration
        self.all_trotter_evolution_data = {}
        self.all_trotter_evolution_error = {}
        time_array = np.linspace(0, total_time, time_steps + 1)
        # Process raw data into DataFrames
        for key, data_dict in observables_raw_data.items():
            df = pd.DataFrame.from_records(data_dict, index=time_array)
            df.index.name = "Time"
            # Unpack tuples/arrays in columns if needed and drop original columns
            self.all_trotter_evolution_data[key] = self.unpack_columns(
                df, drop_original=True
            )
        # Process raw errors into DataFrames
        for key, data_dict in observables_raw_error.items():
            df = pd.DataFrame.from_records(data_dict, index=time_array)
            df.index.name = "Time"
            # Unpack tuples/arrays in columns if needed and drop original columns
            self.all_trotter_evolution_error[key] = self.unpack_columns(
                df, drop_original=True
            )

        # Set one variable as the main observables dataframe
        preferential_key = evolution_params.get("Preferential_Trotter", None)

        # Apply mitigation if specified
        apply_algorithmic_mitigation = evolution_params.get(
            "Algorithmic_Mitigation", {}
        ).get("Apply", False)
        if apply_algorithmic_mitigation and len(trotter_configs) > 1:
            # Get Trotter order (Qiskit default is 2)
            trotter_order = synthesis_params.get("order", 2)
            print(
                f"{getTimer()} INFO: Applying Richardson Extrapolation with Trotter order {trotter_order}..."
            )
            # We need at least 2 different time steps to apply Richardson extrapolation
            coarse_key = max(
                trotter_configs.keys(),
                key=lambda k: trotter_configs[k].get("step_multiplier", 1.0),
            )
            fine_key = min(
                trotter_configs.keys(),
                key=lambda k: trotter_configs[k].get("step_multiplier", 1.0),
            )
            mitigated_data = self.apply_richardson_extrapolation(
                self.all_trotter_evolution_data[coarse_key],
                self.all_trotter_evolution_data[fine_key],
                step_mult_coarse=trotter_configs[coarse_key].get(
                    "step_multiplier", 1.0
                ),
                step_mult_fine=trotter_configs[fine_key].get("step_multiplier", 1.0),
                trotter_order=trotter_order,
            )
            mitigated_error = self.apply_richardson_extrapolation(
                self.all_trotter_evolution_error[coarse_key],
                self.all_trotter_evolution_error[fine_key],
                step_mult_coarse=trotter_configs[coarse_key].get(
                    "step_multiplier", 1.0
                ),
                step_mult_fine=trotter_configs[fine_key].get("step_multiplier", 1.0),
                trotter_order=trotter_order,
            )
            self.all_trotter_evolution_data["mitigated"] = mitigated_data
            self.all_trotter_evolution_error["mitigated"] = mitigated_error
            if preferential_key is None:
                preferential_key = "mitigated"
            state = states[
                fine_key
            ]  # The state evolved with the finest time step is the closest to the mitigated data
        else:
            if preferential_key is None:
                preferential_key = min(
                    trotter_configs.keys(),
                    key=lambda k: trotter_configs[k].get("step_multiplier", 1.0),
                )
            state = states[preferential_key]

        # Main DataFrames
        self.observables_dataframe = self.all_trotter_evolution_data[preferential_key]
        self.observables_dataframe_error = self.all_trotter_evolution_error[
            preferential_key
        ]

        return (
            state,
            self.observables_dataframe,
            self.observables_dataframe_error,
            self.all_trotter_evolution_data,
            self.all_trotter_evolution_error,
        )

    def _initiate_observables(
        self, observables_list: list, observables_params: dict | None = None
    ) -> list[BaseObservable]:
        """
        Initiates the BaseObservable objects of the simulation from the list observables_list.

        Parameters
        ----------
        observables_list: list of str
            List with the names of the observables to calculate during the evolution (e.g. ["Energy", "Gauss_Law_Violation", "Pair_Creation"]).
        observables_params: dict | None, optional
            Dictionary with parameters for each observable, by default None (not implemented for any observable yet).

        Returns
        -------
        active_observables: list of BaseObservable objects
            List with the initiated observable objects corresponding to the names in observables_list.
        """
        active_observables = []
        if observables_params is None:
            observables_params = {}
        for obs_name in observables_list:
            spec_params = observables_params.get(obs_name, {})

            if obs_name == "Energy":
                obs = EnergyObservable(self.hamiltonian_evol, self.precision)

            elif obs_name == "Persistence":
                # Needs initial state
                obs = PersistenceObservable(self.initial_state, self.qubits_num)

            elif obs_name == "Pair_Creation":
                obs = PairCreationObservable(self.qubits_num, self.precision)

            elif obs_name == "Gauss_Law_Violation":
                obs = GaussLawViolationObservable(self.qubits_num, self.precision)

            elif obs_name == "Electric_Field":
                # Needs e0 from quench
                obs = ElectricFieldObservable(self.qubits_num, self.e0, self.precision)

            else:
                print(f"WARNING: Observable {obs_name} not implemented...")
                continue

            active_observables.append(obs)

        return active_observables

    def unpack_columns(
        self, observables_dataframe: pd.DataFrame, drop_original: bool = False
    ) -> pd.DataFrame:
        """
        Create new columns in the observables dataframe for columns that come as tuples or arrays,
        such as "Pair_Creation" or "Electric_Field", for better visualization and analysis.
        If drop_original is True, the original columns with tuples/arrays are dropped after unpacking.

        Parameters
        ----------
        observables_dataframe: pd.DataFrame
            DataFrame with the raw observables data, which may contain columns with tuples or arrays.
        drop_original: bool, optional
            Whether to drop the original columns with tuples/arrays after unpacking, by default False.

        Returns
        -------
        observables_dataframe: pd.DataFrame
            DataFrame with new columns for the unpacked observables, and optionally without the original columns.
        """
        if "Pair_Creation" in observables_dataframe.columns:
            # Two columns, electrons and positrons
            df_pairs = pd.DataFrame(
                observables_dataframe["Pair_Creation"].tolist(),
                index=observables_dataframe.index,
            )
            observables_dataframe["Pair_Creation_electrons"] = df_pairs[0]
            observables_dataframe["Pair_Creation_positrons"] = df_pairs[1]
            observables_dataframe["Pair_Creation_balance"] = df_pairs[0] - df_pairs[1]
            if drop_original:
                observables_dataframe.drop(columns=["Pair_Creation"], inplace=True)

        if "Electric_Field" in observables_dataframe.columns:
            # Column with arrays, to L-1 individual columns
            df_ef = pd.DataFrame(
                observables_dataframe["Electric_Field"].tolist(),
                index=observables_dataframe.index,
            )
            # Name as E_link_0, E_link_1, etc.
            ef_col_names = [f"E_link_{i}" for i in range(df_ef.shape[1])]
            df_ef.columns = ef_col_names

            # Concatenate with original dataframe and drop original column
            observables_dataframe = pd.concat([observables_dataframe, df_ef], axis=1)
            if drop_original:
                observables_dataframe.drop(columns=["Electric_Field"], inplace=True)

        return observables_dataframe

    def apply_richardson_extrapolation(
        self,
        df_coarse: pd.DataFrame,
        df_fine: pd.DataFrame,
        step_mult_coarse: float,
        step_mult_fine: float,
        trotter_order: int,
        is_error: bool = False,
    ) -> pd.DataFrame:
        """
        Applies Richardson Extrapolation to mitigate Trotter errors in the observables dataframes,
        based on the algorithmic order and the ratio between the time steps.

        Parameters
        ----------
        df_coarse: pd.DataFrame
            DataFrame evaluated with the larger time step.
        df_fine: pd.DataFrame
            DataFrame evaluated with the smaller time step.
        step_mult_coarse: float
            Multiplier for the large time step (e.g., 1.0 for dt).
        step_mult_fine: float
            Multiplier for the small time step (e.g., 0.5 for dt/2 or 0.25 for dt/4).
        trotter_order: int
            Order of the Suzuki-Trotter algorithm (typically 1, 2 or 4).
        is_error: bool, optional
            Whether the dataframes are for error estimates (True) or for observable values (False). Default is False. If True, the formula is adapted to combine errors in quadrature.

        Returns
        -------
        df_mitigated: pd.DataFrame
            DataFrame with mitigated observables values after applying Richardson Extrapolation.
        """
        # r is the ratio between the large and small time steps (e.g., 1.0 / 0.5 = 2.0)
        r = step_mult_coarse / step_mult_fine

        # Scale factor for the error based on the algorithmic order of Trotter (r^p)
        factor = r**trotter_order

        # Generic formula for Richardson Extrapolation
        if is_error:
            term_fine = (factor / (factor - 1)) ** 2 * (df_fine**2)
            term_coarse = (1 / (factor - 1)) ** 2 * (df_coarse**2)
            df_mitigated = np.sqrt(term_fine + term_coarse)
        else:
            df_mitigated = (factor * df_fine - df_coarse) / (factor - 1)

        return df_mitigated

    def calculate_observable(
        self,
        observable: str,
        state: Statevector | QuantumCircuit,
        initial_state: Statevector | QuantumCircuit | None = None,
        spec_params: Mapping | None = None,
        estimator: BaseEstimatorV2 | None = None,
        sampler: BaseSamplerV2 | None = None,
        precision: float | None = None,
    ):
        """
        [DEPRECATED]
        Calculate the expectation value of a given observable.

        Parameters:
        - observable: str, name of the observable to calculate (e.g. "Energy", "Persistence", "Gauss_Law_Violation", "Pair_Creation").
        - state: Statevector, the state for which to calculate the observable.
        - initial_state: optional (default: None), Statevector, the initial state of the system (used for some observables like Persistence).
        - spec_params: optional (default: None), Mapping, specific parameters for the observable calculation if needed.
        """

        from QuantumSimulation.Calculations import (
            calculateEnergy,
            calculateVacuumPersistence,
            calculateGaussLawViolation,
            checkChargeSymmetry,
            calculatePairCreation,
            calculateElectricField,
        )
        if observable == "Energy":
            value = calculateEnergy(state, self.hamiltonian_evol, estimator, precision)
        elif observable == "Persistence":
            value = calculateVacuumPersistence(state, initial_state, sampler)
        elif observable == "Gauss_Law_Violation":
            value = calculateGaussLawViolation(
                state, self.qubits_num, estimator, precision
            )
        elif observable == "Pair_Creation":
            value = calculatePairCreation(state, self.qubits_num, estimator, precision)
        elif observable == "Electric_Field":
            value = calculateElectricField(
                state, self.qubits_num, self.e0, estimator, precision
            )
        else:
            print(f"{getTimer()} WARNING: Observable {observable} not implemented...")
            value = None

        return value

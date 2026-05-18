# Main file for Schwinger simulation

import sys
import numpy as np
import pandas as pd
import time
import copy
from typing import Callable, Any, Mapping, Iterable
from tqdm.auto import tqdm
from qiskit.circuit.quantumcircuit import QuantumCircuit
from circuitBuilder import buildCircuit, addGate
from Operators import buildSchwingerHamiltonianTemporalGauge
from Utils import getTimer, func_return
from Calculations import calculateEnergy, calculateVacuumPersistence, calculateGaussLawViolation, checkChargeSymmetry, calculatePairCreation, calculateElectricField
from Calculations import EnergyObservable, PersistenceObservable,\
    PairCreationObservable, GaussLawViolationObservable, ElectricFieldObservable
from Ansatzes import build_schwinger_hva, build_schwinger_hva_full, build_schwinger_hva_balanced
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit.library import efficient_su2, n_local, excitation_preserving
from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.primitives import BaseEstimatorV2, BaseSamplerV2, StatevectorEstimator, StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply

class SchwingerSimulation:
    '''
    Main class for Schwinger simulation.

    It takes a configuration dictionary as input and runs the simulation according to the parameters specified in the configuration.

    It builds a lattice hamiltonian, finds the vacuum state with a VQE-like optimization, and optionally then evolves the state in time according to the specified temporal evolution parameters, calculating observables along the way.

    Full simulation is performed by calling the run_simulation() method.

    When initialized, it only stores the configuration and prepares the backend configuration if specified.

    Parameters in simulation_config:
    - "QubitsNumber": int, number of qubits in the simulation (Lattice size).
    - "Hamiltonian": dict, parameters for the initial hamiltonian construction.
    - "Ansatz": dict, parameters for the ansatz construction and vacuum state optimization.
    - "Temporal Evolution": dict, parameters for the temporal evolution (optional).
    - "Backend": dict, parameters for the quantum backend (optional).

    Workflow:
    1. Initialize with configuration.
    2. run_simulation() method performs:

       a. Build initial hamiltonian according to parameters, calling self.get_hamiltonian().

       b. Check symmetries and add charge penalty if specified.

       c. Build ansatz circuit according to parameters, calling self.get_ansatz().

       d. Optimize ansatz parameters to find vacuum state, obtaining the initial state and vacuum energy, calling self.get_vacuum().

       e. If temporal evolution is active, evolve the state according to the specified parameters, calling self.evolve_state().
    '''
    def __init__(self,
                 simulation_config: Mapping[str, Any],
                 initial_state: Statevector | None = None):
        
        print("\n" + "#" * 70)
        print(f"{getTimer()} INFO: Initializing SchwingerSimulation class.")
        # Configuration parameters
        self.config        = simulation_config
        # Number of qubits
        self.qubits_num    = self.config["QubitsNumber"]
        # Initial state (if given, it will be used instead of optimizing for the vacuum state)
        self.initial_state = initial_state
        # Get sampler and estimator based on backend configuration if provided
        self.backend, self.estimator, self.sampler  = self.setup_quantum_primitives()

    def run_simulation(self):

        print(f"{getTimer()} INFO: Starting simulation.")
        ####################
        # 1. Get main params for the simulation
        ####################
        # Get hamiltonian (t < 0 for later quench)
        print(f"{getTimer()} INFO: Parameters of the hamiltonian: {self.config['Hamiltonian']['Parameters']}")
        self.hamiltonian_prep = self.get_hamiltonian()
        if self.hamiltonian_prep is None: return

        # Check symmetries
        hasSym, Q_op = checkChargeSymmetry(self.hamiltonian_prep)
        if not hasSym: return

        # Add charge penalty in order to minimize charge symmetry violations
        lambda_penalty = self.config["Hamiltonian"].get("Lambda_Charge_Penalty", 0)
        if lambda_penalty == "Variable":
            max_coef = self.hamiltonian_prep.coeffs.max()
            lambda_penalty = max_coef * 10
            print(f"{getTimer()} INFO: Lambda_Charge_Penalty set to variable value: {lambda_penalty:.2e} (10 times the max coefficient of the Hamiltonian)")

        if lambda_penalty > 0:
            print(f"{getTimer()} INFO: Adding charge penalty term with lambda = {lambda_penalty}")
            self.hamiltonian_prep = (self.hamiltonian_prep + lambda_penalty * Q_op.dot(Q_op)).simplify()

        # Get ansatz from config:
        # Initial config and type
        self.ansatz = self.get_ansatz()

        # ISA Transpilation for primitives V2
        # Transpile if we use a real estimator or a simulator with backend
        if self.estimator is not None and self.backend is not None:
            print(f"{getTimer()} INFO: Transpiling ansatz to ISA circuit...")
            
            # Generate PassManager with the specified backend (AerSimulator or IBM Runtime)
            pm = generate_preset_pass_manager(optimization_level=1, backend=self.backend)
            
            # Transpile ansatz to the target backend's ISA using the PassManager
            self.ansatz = pm.run(self.ansatz)
            
            # When transpiling, manager can reorder physical qubits
            # Hamiltonian topology must be adapted
            if self.ansatz.layout is not None:
                self.hamiltonian_prep = self.hamiltonian_prep.apply_layout(self.ansatz.layout)

        # Get parameters that minimize initial ansatz energy
        if self.initial_state is None:
            self.initial_state, self.vacuum_energy, self.vacuum_parameters = self.get_vacuum()
        else:
            self.vacuum_energy     = calculateEnergy(self.initial_state, self.hamiltonian_prep, self.estimator, self.precision)
            self.vacuum_parameters = None

        print(f"{getTimer()} INFO: Initial energy = {self.vacuum_energy}")
        
        # Ground state diagnostics
        if self.config["Ansatz"].get("Ensure_Zero_Charge", False):
            Q_total_inicial = Statevector(self.initial_state).expectation_value(Q_op).real
            print(f"{getTimer()} INFO: Initial charge = {Q_total_inicial}")
            if abs(Q_total_inicial) > 0.01:
                print(f"{getTimer()} WARNING: Initial state does not have zero charge, review ansatz and minimization parameters.")
                return
        
        # Store diagnostics for later inspection
        self.ground_state_diagnostics = {
            'vacuum_energy': self.vacuum_energy,
            'vacuum_charge': Q_total_inicial if self.config["Ansatz"].get("Ensure_Zero_Charge", False) else None,
            'optimization_history': self.optimization_history if hasattr(self, 'optimization_history') else []
        }
        
        ####################
        # 2. Temporal evolution
        ####################

        if self.config.get("Temporal Evolution", {}).get("Active", False):
            self.final_state, self.evolution_data, self.all_trotter_evolution_data = self.evolve_state()

        # Additional results and calculations (not developed in this module)
        # Check Results for results examples
        print(f"{getTimer()} INFO: Simulation ended.")
        print("#" * 70 + "\n")

    def setup_quantum_primitives(self):
        '''
        Initializes and returns the Backend and the appropriate Qiskit V2 Primitives (Estimator and Sampler)
        based on the backend configuration. Supports Aer simulators and real IBM Quantum hardware.
        '''
        backend_config = self.config.get("Backend", {})
        if not backend_config:
            self.backend_type = None
            self.precision = None
            print(f"{getTimer()} INFO: No backend specified. Evolving with direct matrix operations.")
            return None, None, None

        self.backend_type = backend_config.get("Type", "Aer")
        backend_options   = backend_config.get("Options", {})
        self.precision    = backend_options.get("Precision", None)
        shots             = backend_options.get("shots", 1024)

        if self.backend_type == "StatevectorEstimator":
            print(f"{getTimer()} INFO: Using StatevectorEstimator (Ideal V2 Primitive).")
            return None, StatevectorEstimator(), StatevectorSampler()
            
        elif self.backend_type in ["Aer", "AerSimulator"]:
            print(f"{getTimer()} INFO: Using Primitives V2 backed by AerSimulator.")
            aer_backend = AerSimulator(**backend_options)
            aer_options = {
                "backend_options": backend_options, # Handles noise_model, coupling_map, etc.
                "run_options": {"shots": shots}     # Handles shots for the Estimator
            }
            estimator = EstimatorV2(options=aer_options)
            sampler   = SamplerV2(options=aer_options)
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
            sampler   = RuntimeSamplerV2(mode=backend)
            
            # Apply resilience levels if specified
            estimator.options.resilience_level = backend_options.get("resilience_level", 1)
            
            return backend, estimator, sampler
            
        else:
            print(f"{getTimer()} WARNING: Backend type {self.backend_type} not recognized. Reverting to exact multiplication.")
            return None, None, None
        
    def get_hamiltonian(self,
                        override_params: dict | None =None):
        '''
        Get hamiltonian according to the entry "Hamiltonian" in self.config.

        If override_params is given, it overwrites the parameters in the configuration for the hamiltonian construction. This is useful for quenches in the temporal evolution.

        Initialized class variables needed: self.config, self.qubits_num

        Parameters in self.config["Hamiltonian"]:
        - "Type": str, type of hamiltonian to build (e.g. "Schwinger").
        - "Gauge": str, gauge to use for the hamiltonian construction. Default "Temporal".
        - "Parameters": dict, parameters for the hamiltonian construction. Lattice parameters:
            - "L": int, lattice size (number of sites).
            - "m": float, fermion mass.
            - "a": float, lattice spacing.
            - "e0": optional, float, background field.

        Returns:
        - hamiltonian: SparsePauliOp, the hamiltonian operator in the form of a sparse Pauli sum if configuration is implemented, else None.
        '''
        ham_type   = self.config["Hamiltonian"]["Type"]
        ham_params = self.config["Hamiltonian"]["Parameters"].copy()
        ham_gauge  = self.config["Hamiltonian"].get("Gauge", "Temporal")

        # Overwrite quench params if applies
        if override_params:
            ham_params.update(override_params)

        if ham_type == "Schwinger":
            if ham_gauge == "Temporal":
                return func_return(buildSchwingerHamiltonianTemporalGauge, ham_params, expect_type=SparsePauliOp)
            else:
                print(f"{getTimer()} WARNING: Gauge {ham_gauge} not implemented...")
                return None
        else:
            print(f"{getTimer()} WARNING: Hamiltonian type {ham_type} not implemented...")
            return None

    def get_ansatz(self):
        '''
        Build initial ansatz circuit according to the entry "Ansatz" in self.config.

        Initialized class variables needed: self.config, self.qubits_num

        Parameters in self.config["Ansatz"]:
        - "Type": str, type of ansatz to build (e.g. "EfficientSU2", "ExcitationPreserving", "TwoLocal").
        - "Initial State": optional, dict, parameters for the initial state preparation. Initial state parameters:
            - "Vacuum": bool (default: True), whether to prepare the vacuum state as initial state.
            - "Staggered": bool (default: True), whether to use a staggered configuration for the initial state (electrons in even sites and positrons in odd sites).
        - "Entanglement": str (default: "linear"), type of entanglement for the ansatz (e.g. "linear", "full", etc.).
        - "Reps": int (default: 3), number of repetitions for the ansatz layers.
        - "AdditionalParams": dict (default: {}), additional parameters for the ansatz construction if needed.
        '''
        ansatz_type     = self.config["Ansatz"]["Type"]
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
                print(f"{getTimer()} WARNING: Ansatz state for not staggered fermions not implemented...")
                return None
        else:
            print(f"{getTimer()} WARNING: Ansatz state for not vacuum state not implemented...")
            return None

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
                    **{"hamiltonian": self.hamiltonian_prep}
                }
            else:
                self.config["Ansatz"]["AdditionalParams"] = {"hamiltonian": self.hamiltonian_prep}
        elif ansatz_type == "HVA Full":
            ansatz_func = build_schwinger_hva_full
            if self.config["Ansatz"].get("AdditionalParams", {}):
                self.config["Ansatz"]["AdditionalParams"] = {
                    **self.config["Ansatz"]["AdditionalParams"],
                    **{"hamiltonian": self.hamiltonian_prep}
                }
            else:
                self.config["Ansatz"]["AdditionalParams"] = {"hamiltonian": self.hamiltonian_prep}
        elif ansatz_type == "HVA Simple":
            ansatz_func = build_schwinger_hva
            if self.config["Ansatz"].get("AdditionalParams", {}):
                self.config["Ansatz"]["AdditionalParams"] = {
                    **self.config["Ansatz"]["AdditionalParams"],
                    **{"hamiltonian": self.hamiltonian_prep}
                }
            else:
                self.config["Ansatz"]["AdditionalParams"] = {"hamiltonian": self.hamiltonian_prep}
        else:
            print(f"{getTimer()} WARNING: Ansatz type {ansatz_type} not implemented...")
            return None
        
        ansatz_params = {
            "num_qubits": self.qubits_num,
            "entanglement": self.config["Ansatz"].get("Entanglement", "linear"),
            "reps": self.config["Ansatz"].get("Reps", 3)
        }

        print(f"{getTimer()} INFO: Using ansatz {ansatz_type} with parameters: {ansatz_params}")

        ansatz_params = {
            **ansatz_params,
            **self.config["Ansatz"].get("AdditionalParams", {})
        }

        ansatz = func_return(ansatz_func, ansatz_params)
        assert isinstance(ansatz, QuantumCircuit), f"WARNING: Ansatz function did not return a QuantumCircuit, review ansatz parameters..."
        if ansatz is not None:
            # Evolve from initial vacuum
            full_ansatz_circuit = initial_circuit.compose(ansatz)
        return full_ansatz_circuit

    def get_vacuum(self):
        '''
        Optimize ansatz parameters to find the vacuum state, minimizing the energy cost function, according to parameters in self.config["Ansatz"].

        Initialized class variables needed: self.config, self.ansatz (from self.get_ansatz()), self.hamiltonian_prep (from self.get_hamiltonian())

        Parameters in self.config["Ansatz"]:
        - "Minimizer": dict, parameters for the minimization algorithm. Minimizer parameters:
            - "Method": str, optimization method to use (e.g. "COBYLA", "Nelder-Mead", "BFGS", etc.). Default: "COBYLA".
            - "Options": optional, dict, options for the optimization method (e.g. {"maxiter": 1000, "tol": 1e-6}). Default: {"maxiter": 2000, "tol": 1e-6}.
            - "AdditionalParams": optional, dict, additional parameters for the optimization method if needed.
        - "Initial Parameters": optional, array-like, initial parameters for the optimization. If not given, it is initialized randomly (with a fixed seed for reproducibility). Initial parameters strategy can be configured with "Init_Strategy" (default: "random_small", which initializes parameters with small random values near zero, better for convergence in VQE-like optimizations).
        - "Init_Strategy": optional, str, strategy for initializing parameters if "Initial Parameters" is not given. Options: "random_small", "uniform_random". Default: "random_small".

        Returns:
        - vacuum_state: Statevector, the statevector of the optimized vacuum state.
        - vacuum_energy: float, the energy of the optimized vacuum state.
        - vacuum_parameters: array-like, the parameters of the ansatz that minimize the energy cost function.
        '''
        # Assert needed variables are defined
        assert hasattr(self, 'ansatz') and self.ansatz is not None, f"Ansatz not defined, cannot optimize vacuum state. Make sure to call get_ansatz() before get_vacuum() or assign it explicitly."
        assert hasattr(self, 'hamiltonian_prep') and self.hamiltonian_prep is not None, f"Hamiltonian not defined, cannot optimize vacuum state. Make sure to call get_hamiltonian() before get_vacuum() or assign it explicitly."

        initial_state_params = self.config["Ansatz"].get("Initial Parameters", None)
        if not initial_state_params:
            #np.random.seed(42)
            # IMPROVED: Better parameter initialization (near identity instead of uniform random)
            init_strategy = self.config["Ansatz"].get("Init_Strategy", "random_small")
            init_max      = self.config["Ansatz"].get("Init_Max", 0.1)
            if init_strategy == "random_small":
                # Start near identity: better convergence for VQE
                initial_state_params = np.random.normal(0, init_max, self.ansatz.num_parameters)
            elif init_strategy == "zeros":
                # Start near identity: better convergence for VQE
                initial_state_params = np.zeros(self.ansatz.num_parameters)
            elif init_strategy == "uniform_random":
                # Original uniform random
                initial_state_params = np.random.random(self.ansatz.num_parameters) * 2 * np.pi
            else:
                # Default to small random
                initial_state_params = np.random.normal(0, 0.1, self.ansatz.num_parameters)

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
            **self.config["Ansatz"]["Minimizer"].get("AdditionalParams", {})
        }

        if self.config["Ansatz"].get("Use Gradient Cost", False):
            if hasattr(self, "estimator"): self.gradient_estimator = self.estimator
            else:                          self.gradient_estimator = StatevectorEstimator()
            minimize_params["jac"] = self.gradient_cost_function
            print(f"{getTimer()} INFO: Analytic gradient (ParamShift) enabled.")

        print(f"{getTimer()} INFO: Minimization config: method={minimize_params['method']}, maxiter={final_options['maxiter']}")
        print(f"{getTimer()} INFO: Initial parameters strategy: {init_strategy}")
        
        # IMPROVED: Track optimization history for diagnostics
        self.optimization_history = []

        def callback(xk):
            energy = self.energy_cost_function(xk)
            self.optimization_history.append(energy)
            # Update progress bar
            try:    pbar.update(1)
            except: pass

        max_iter = final_options.get('maxiter', 2000)
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
            
            print(f"{getTimer()} INFO: Opening IBM Runtime Session for VQE optimization...")
            # Open exclusive VQE session
            with Session(backend=self.backend) as session:
                # New estimator for this session
                session_estimator = RuntimeEstimatorV2(mode=session)
                session_estimator.options.resilience_level = self.estimator.options.resilience_level
                
                # We change the original estimator so the Session uses the one associated with it
                original_estimator = self.estimator
                self.estimator     = session_estimator
                
                # Launch VQE
                with tqdm(total=max_iter, desc="VQE Optimization (Session)", unit="iter", file=sys.stdout) as pbar:
                    result = minimize(self.energy_cost_function, initial_state_params, **minimize_params, callback=callback)
                    if hasattr(result, 'nit'):
                        pbar.total = pbar.n = result.nit
                        pbar.refresh()
                        
                # Get back original estimator for the rest of the simulation
                self.estimator = original_estimator
                print(f"{getTimer()} INFO: IBM Runtime Session closed successfully.")
        else:
            with tqdm(total=max_iter, desc="VQE Optimization", unit="iter", file=sys.stdout, leave=True, dynamic_ncols=False) as pbar:
                result = minimize(self.energy_cost_function, initial_state_params, **minimize_params,
                                callback=callback)
                # Update bar to show actual iterations
                if hasattr(result, 'nit'):
                    pbar.total = result.nit
                    pbar.n = result.nit
                    pbar.refresh()

        self.vqe_duration = time.time() - vqe_start
        self.vqe_iters    = result.nit if hasattr(result, 'nit') else None
        self.vqe_success  = result.success

        vacuum_parameters = result.x
        vacuum_state      = self.ansatz.assign_parameters(vacuum_parameters)
        vacuum_energy     = result.fun
        
        # IMPROVED: Diagnostic information
        print(f"{getTimer()} INFO: Optimization completed.")
        print(f"{getTimer()} INFO:   Converged: {result.success}")
        print(f"{getTimer()} INFO:   Iterations: {result.nit if hasattr(result, 'nit') else 'N/A'}")
        print(f"{getTimer()} INFO:   Final energy: {vacuum_energy:.8f}")
        if len(self.optimization_history) > 1:
            energy_improvement = self.optimization_history[0] - self.optimization_history[-1]
            print(f"{getTimer()} INFO:   Energy improvement: {energy_improvement:.8f}")
        
        return vacuum_state, vacuum_energy, vacuum_parameters

    def energy_cost_function(self, params: Mapping | Iterable):
        '''
        Cost funcion to minimize energy of the state
        obtained by the ansatz with given parameters,
        given self.hamiltonian_prep and self.ansatz.
        '''
        ansatz_circuit = self.ansatz.assign_parameters(params)
        if hasattr(self, 'estimator') and self.estimator is not None:
            pub = self._energy_cost.get_pub(ansatz_circuit)
            result = self.estimator.run([pub]).result()
            return self._energy_cost.process_pub_result(result[0])
        else:
            return self._energy_cost.calculate_exact(ansatz_circuit)
    
    def gradient_cost_function(self, params: Mapping | Iterable):
        '''
        Calculate analytical gradient using Parameter Shift Rule.
        '''
        # Gradient calculator initialization
        gradient = ParamShiftEstimatorGradient(estimator=self.gradient_estimator)
        
        # Run job
        job = gradient.run([self.ansatz], [self.hamiltonian_prep], [params])
        
        return np.array(job.result().gradients[0])    
    
    def evolve_state(self):
        '''
        Evolves the initial state with the configuration "Temporal Evolution" at self.config

        Initialized class variables needed: self.config, self.initial_state (from self.get_vacuum()), self.hamiltonian_prep (from self.get_hamiltonian())

        Parameters in self.config["Temporal Evolution"]:
        - "Active": bool, whether to perform temporal evolution or not.
        - "Time_Steps": int, number of time steps for the evolution.
        - "Total_Time": float, total time for the evolution.
        - "Evolution_Gate": dict, parameters for the evolution gate construction. Evolution gate parameters:
            - "Type": str, type of evolution gate to use (e.g. "Pauli"). Default: "Pauli".
            - "Synthesis": str, method for the evolution gate synthesis (e.g. "TrotterSuzuki"). Default: "TrotterSuzuki".
            - "Synthesis_Params": dict, additional parameters for the evolution gate synthesis if needed.
        - "Quench": dict, parameters for a possible quench in the evolution. Quench parameters:
            - "Active": bool, whether to perform a quench or not.
            - "Parameters_to_Change": dict, parameters to change for the quench in the hamiltonian construction (e.g. {"m": 0.5} to change the fermion mass to 0.5 after the quench).
        - "Observables": dict, parameters for the observables to calculate during the evolution. Observables parameters:
            - "Observables_List": list of str, list of observables to calculate (e.g. ["Energy", "Gauss_Law_Violation", "Pair_Creation"]).
            - "Observables_Params": dict, additional parameters for the observables if needed (e.g. for "Pair_Creation", {"Lattice_Size": L} to specify the lattice size for the pair creation calculation).

        Returns: 
         - state: Statevector of the final state of the evolution.

         - observables_dataframe: DataFrame with time values of the evolution as index and columns the observables given in the list in the configuration for the best Trotter configuration.

         - trotter_evolution_data: dict with the evolution data for each Trotter configuration if "Trotter_Steps" is given in the configuration
        '''

        # Assert needed variables are defined
        assert hasattr(self, 'initial_state') and self.initial_state is not None, f"Initial state not defined, cannot evolve state. Make sure to call get_vacuum() before evolve_state() or assign it explicitly."
        assert hasattr(self, 'hamiltonian_prep') and self.hamiltonian_prep is not None, f"Hamiltonian not defined, cannot optimize vacuum state. Make sure to call get_hamiltonian() before get_vacuum() or assign it explicitly."

        evolution_params = self.config["Temporal Evolution"]

        time_steps = evolution_params["Time_Steps"]
        total_time = evolution_params["Total_Time"]
        step = total_time / time_steps

        evolution_gate_params = evolution_params["Evolution_Gate"]
        evolution_gate_type   = evolution_gate_params.get("Type", "Pauli")
        if evolution_gate_type == "Pauli":
            gate = PauliEvolutionGate
        else:
            print(f"{getTimer()} WARNING: Evolution gate type {evolution_gate_type} not implemented...")
            return 
        synthesis_method = evolution_gate_params.get("Synthesis", "TrotterSuzuki")
        synthesis_params = evolution_gate_params.get("Synthesis_Params", {})
        if synthesis_method == "TrotterSuzuki":
            synthesis = SuzukiTrotter
        else:
            print(f"{getTimer()} WARNING: Synthesis method {synthesis_method} not implemented...")
            return 
        
        print(f"{getTimer()} INFO: Evolution gate type {evolution_gate_type} with synthesis method {synthesis_method} selected. Synthesis parameters: {synthesis_params}")

        quench_config = evolution_params.get("Quench", {})
        if quench_config.get("Active", False):
            quench_params = quench_config.get("Parameters_to_Change", {})
            print(f"{getTimer()} INFO: Applying Quench with parameters: {quench_params}")
            self.hamiltonian_evol = self.get_hamiltonian(override_params=quench_params)
            self.e0 = quench_params.get("e0", self.config["Hamiltonian"]["Parameters"].get("e0", 0))
        else:
            print(f"{getTimer()} INFO: No Quench parameters, evolving with base hamiltonian.")
            self.hamiltonian_evol = self.hamiltonian_prep
            self.e0 = self.config["Hamiltonian"]["Parameters"].get("e0", 0)
        
        
        # Define initial state
        if self.estimator is None:
            state = Statevector.from_instruction(self.initial_state)
            initial_state = state.copy()
        else:
            state = self.initial_state.copy()
            initial_state = state.copy()
            estimator_pubs = {"pubs": [], "info": [], "trotter_key": []}
            sampler_pubs   = {"pubs": [], "info": [], "trotter_key": []}

        # Get Trotter configurations if defined for different time steps
        trotter_configs = evolution_params.get("Trotter_Steps", None)
        if not trotter_configs: trotter_configs = {"dt": {"step_multiplier": 1.0}}
        trotter_circuits = {}
        states = {}

        for key, t_conf in trotter_configs.items():
            multiplier        = t_conf.get("step_multiplier", 1.0)
            current_time_step = step * multiplier
            evolution_gate    = gate(self.hamiltonian_evol, time=current_time_step,
                                     synthesis=synthesis(**synthesis_params))
            
            # Construct Trotter circuit to compose
            if self.estimator is not None:
                trotter_step_circuit = QuantumCircuit(self.qubits_num)
                trotter_step_circuit.append(evolution_gate, range(self.qubits_num))
                
                if self.backend is not None and self.backend_type not in ["StatevectorEstimator", None]:
                    
                    print(f"{getTimer()} INFO: Transpiling Trotter step to ISA...")
                    
                    # Force PassManager to use same physical layout than VQE
                    layout = self.initial_state.layout if hasattr(self.initial_state, 'layout') else None
                    pm_trotter = generate_preset_pass_manager(optimization_level=1, backend=self.backend, initial_layout=layout)
                    
                    trotter_circuits[key] = pm_trotter.run(trotter_step_circuit)
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
        evolution_method = evolution_params.get("Evolution_Method", "MatrixExponential")

        # Prepare evolution data structures according to backend
        if evolution_method == "MatrixExponential":
            sparse_ham = self.hamiltonian_evol.to_matrix(sparse=True)

        # Observables config dict
        observables        = evolution_params.get("Observables", {})
        # List of all observables
        observables_list   = observables.get("Observables_List", [])
        # Dict with spec parameters for observables {"obs": {"param1": value1, ...}, ...}
        observables_params = observables.get("Observables_Params", {})
        # Initialization of observable object of type BaseObservable
        observables_objects = self._initiate_observables(observables_list, observables_params)
        # Data structures to store observables data during evolution
        observables_data   = {obs: [] for obs in observables_list}
        observables_raw_data = {
            key: {obs.name: [] for obs in observables_objects} 
            for key in trotter_configs.keys()
        }
        # Iterate over time steps
        with tqdm(range(time_steps + 1), desc="Evolving state", unit="step", file=sys.stdout, leave=True, dynamic_ncols=False) as pbar:
            for t in pbar: # Time iteration
                for trotter_key, state in list(states.items()):
                    # Calculate observables for current state
                    for obs in observables_objects:
                        if self.estimator is None:
                            value = obs.calculate_exact(state)
                            observables_raw_data[trotter_key][obs.name].append(value)
                        else:
                            pub = obs.get_pub(state)
                            if obs.primitive_type == "estimator":
                                estimator_pubs["pubs"]        += [pub]
                                estimator_pubs["info"]        += [obs]
                                estimator_pubs["trotter_key"] += [trotter_key]
                            elif obs.primitive_type == "sampler":
                                sampler_pubs["pubs"]        += [pub]
                                sampler_pubs["info"]        += [obs]
                                sampler_pubs["trotter_key"] += [trotter_key]

                    # Evolve state
                    if t < time_steps: # No need to evolve at the last step
                        multiplier = trotter_configs[trotter_key].get("step_multiplier", 1.0)
                        num_applications = int(1 / multiplier)
                        trotter_step = step * multiplier
                        for _ in range(num_applications):
                            if self.estimator is None:
                                # Evolve state directly (no Aer backend)
                                if evolution_method == "MatrixExponential":
                                    state_data = expm_multiply(-1j * sparse_ham * trotter_step, state.data)
                                    state = Statevector(state_data)
                                else:
                                    # Default: use gate evolution (slower but exact)
                                    state = state.evolve(trotter_circuits[trotter_key])
                            else:
                                state.compose(trotter_circuits[trotter_key], range(self.qubits_num), inplace=True)
                        
                        # Set evolved state for next Trotter step
                        states[trotter_key] = state
        
        # Process PUBS results if using estimator/sampler
        if self.estimator is not None:
            print(f"{getTimer()} INFO: Running {len(estimator_pubs['pubs'])} Estimator PUBS...")
            res_est = self.estimator.run(estimator_pubs['pubs']).result()
            for result, obs_obj, trotter_key in zip(res_est, estimator_pubs['info'], estimator_pubs['trotter_key']):
                observables_raw_data[trotter_key][obs_obj.name].append(obs_obj.process_pub_result(result))
                
            if sampler_pubs["pubs"]:
                print(f"{getTimer()} INFO: Running {len(sampler_pubs['pubs'])} Sampler PUBS...")
                res_samp = self.sampler.run(sampler_pubs['pubs']).result()
                for result, obs_obj, trotter_key in zip(res_samp, sampler_pubs['info'], sampler_pubs['trotter_key']):
                    observables_raw_data[trotter_key][obs_obj.name].append(obs_obj.process_pub_result(result))
        
        # Dict to store DataFrames for each Trotter configuration
        self.all_trotter_evolution_data = {}
        time_array = np.linspace(0, total_time, time_steps + 1)
        for key, data_dict in observables_raw_data.items():
            df = pd.DataFrame.from_records(data_dict, index=time_array)
            df.index.name = "Time"
            # Unpack tuples/arrays in columns if needed and drop original columns
            self.all_trotter_evolution_data[key] = self.unpack_columns(df, drop_original=True)      

        # Apply mitigation if specified
        apply_algorithmic_mitigation = evolution_params.get("Algorithmic_Mitigation", {}).get("Apply", False)
        if apply_algorithmic_mitigation and len(trotter_configs) > 1:
            # Get Trotter order (Qiskit default is 2)
            trotter_order = synthesis_params.get("order", 2)
            print(f"{getTimer()} INFO: Applying Richardson Extrapolation with Trotter order {trotter_order}...")        
            # We need at least 2 different time steps to apply Richardson extrapolation
            coarse_key = max(trotter_configs.keys(), key=lambda k: trotter_configs[k].get("step_multiplier", 1.0))
            fine_key   = min(trotter_configs.keys(), key=lambda k: trotter_configs[k].get("step_multiplier", 1.0))
            mitigated_data = self.apply_richardson_extrapolation(self.all_trotter_evolution_data[coarse_key],
                                                                self.all_trotter_evolution_data[fine_key],
                                                                step_mult_coarse=trotter_configs[coarse_key].get("step_multiplier", 1.0),
                                                                step_mult_fine=trotter_configs[fine_key].get("step_multiplier", 1.0),
                                                                trotter_order=trotter_order)            
            self.all_trotter_evolution_data["mitigated"] = mitigated_data
            preferential_key = "mitigated"
            state = states[fine_key] # The state evolved with the finest time step is the closest to the mitigated data
        else:
            # Set one variable as the main observables dataframe
            preferential_key = evolution_params.get("Preferential_Trotter", None)
            if preferential_key is None:
                preferential_key = min(trotter_configs.keys(), key=lambda k: trotter_configs[k].get("step_multiplier", 1.0))
            state = states[preferential_key]

        self.observables_dataframe = self.all_trotter_evolution_data[preferential_key]

        return state, self.observables_dataframe, self.all_trotter_evolution_data


    def _initiate_observables(self,
                              observables_list: list,
                              observables_params: dict | None = None) -> list:
        '''
        Initiates the BaseObservable objects of the simulation from the list observables_list.
        '''
        active_observables = []
        if observables_params is None: observables_params = {}
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
    
    def unpack_columns(self,
                       observables_dataframe: pd.DataFrame,
                       drop_original: bool = False) -> pd.DataFrame:
        '''
        Create new columns in the observables dataframe for columns that come as tuples or arrays,
        such as "Pair_Creation" or "Electric_Field", for better visualization and analysis.
        If drop_original is True, the original columns with tuples/arrays are dropped after unpacking.
        '''
        if "Pair_Creation" in observables_dataframe.columns:
            # Two columns, electrons and positrons
            df_pairs = pd.DataFrame(observables_dataframe["Pair_Creation"].tolist(), index=observables_dataframe.index)
            observables_dataframe["Pair_Creation_electrons"] = df_pairs[0]
            observables_dataframe["Pair_Creation_positrons"] = df_pairs[1]
            observables_dataframe["Pair_Creation_balance"]   = df_pairs[0] - df_pairs[1]
            if drop_original: observables_dataframe.drop(columns=["Pair_Creation"], inplace=True)
            
        if "Electric_Field" in observables_dataframe.columns:
            # Column with arrays, to L-1 individual columns
            df_ef = pd.DataFrame(observables_dataframe["Electric_Field"].tolist(), index=observables_dataframe.index)
            # Name as E_link_0, E_link_1, etc.
            ef_col_names = [f"E_link_{i}" for i in range(df_ef.shape[1])]
            df_ef.columns = ef_col_names
            
            # Concatenate with original dataframe and drop original column
            observables_dataframe = pd.concat([observables_dataframe, df_ef], axis=1)
            if drop_original: observables_dataframe.drop(columns=["Electric_Field"], inplace=True)

        return observables_dataframe
    
    def apply_richardson_extrapolation(self, 
                                       df_coarse: pd.DataFrame, 
                                       df_fine: pd.DataFrame,
                                       step_mult_coarse: float, 
                                       step_mult_fine: float,
                                       trotter_order: int) -> pd.DataFrame:
        '''
        Applies Richardson Extrapolation to mitigate Trotter errors in the observables dataframes,
        based on the algorithmic order and the ratio between the time steps.
        
        Parameters:
        - df_coarse: DataFrame evaluated with the larger time step.
        - df_fine: DataFrame evaluated with the smaller time step.
        - step_mult_coarse: Multiplier for the large time step (e.g., 1.0 for dt).
        - step_mult_fine: Multiplier for the small time step (e.g., 0.5 for dt/2 or 0.25 for dt/4).
        - trotter_order: Order of the Suzuki-Trotter algorithm (typically 1, 2 or 4).
        '''
        # r is the ratio between the large and small time steps (e.g., 1.0 / 0.5 = 2.0)
        r = step_mult_coarse / step_mult_fine
        
        # Scale factor for the error based on the algorithmic order of Trotter (r^p)
        factor = r ** trotter_order
        
        # Generic formula for Richardson Extrapolation
        df_mitigated = (factor * df_fine - df_coarse) / (factor - 1)
        
        return df_mitigated    
    
    def calculate_observable(self, observable: str,
                             state: Statevector | QuantumCircuit,
                             initial_state: Statevector | QuantumCircuit | None = None,
                             spec_params: Mapping | None = None,
                             estimator: BaseEstimatorV2 | None = None,
                             sampler: BaseSamplerV2 | None = None,
                             precision: float | None = None
                             ):
        '''
        [DEPRECATED]
        Calculate the expectation value of a given observable.
        
        Parameters:
        - observable: str, name of the observable to calculate (e.g. "Energy", "Persistence", "Gauss_Law_Violation", "Pair_Creation").
        - state: Statevector, the state for which to calculate the observable.
        - initial_state: optional (default: None), Statevector, the initial state of the system (used for some observables like Persistence).
        - spec_params: optional (default: None), Mapping, specific parameters for the observable calculation if needed.
        '''
        
        if observable == "Energy":
            value = calculateEnergy(state, self.hamiltonian_evol, estimator, precision)
        elif observable == "Persistence":
            value = calculateVacuumPersistence(state, initial_state, sampler)
        elif observable == "Gauss_Law_Violation":
            value = calculateGaussLawViolation(state, self.qubits_num, estimator, precision)
        elif observable == "Pair_Creation":
            value = calculatePairCreation(state, self.qubits_num, estimator, precision)
        elif observable == "Electric_Field":
            value = calculateElectricField(state, self.qubits_num, self.e0, estimator, precision)
        else:
            print(f"{getTimer()} WARNING: Observable {observable} not implemented...")
            value = None
        
        return value
    

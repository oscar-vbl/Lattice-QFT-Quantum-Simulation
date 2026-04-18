# Main file for Schwinger simulation

import sys
import numpy as np
import pandas as pd
from typing import Callable, Any, Mapping, Iterable
from tqdm.auto import tqdm
from qiskit.circuit.quantumcircuit import QuantumCircuit
from circuitBuilder import buildCircuit, addGate
from Operators import buildSchwingerHamiltonianTemporalGauge
from Utils import getTimer, func_return
from Calculations import calculateEnergy, calculateVacuumPersistence, calculateGaussLawViolation, checkChargeSymmetry, calculatePairCreation
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit.library import efficient_su2, n_local, excitation_preserving
from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.primitives import BaseEstimatorV2, BaseSamplerV2, StatevectorEstimator, StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2
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
        self.estimator, self.sampler  = self.get_backend()

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
            self.final_state, self.evolution_data = self.evolve_state()

        # Plot data (not developed yet, at very first stage)
        # Check ResultsSimulation for simple results examples
        print(f"{getTimer()} INFO: Simulation ended.")
        print("#" * 70 + "\n")

    def get_backend(self):
        backend_config  = self.config.get("Backend", {})
        if backend_config:
            self.backend_type = backend_config.get("Type", "Aer")
            backend_options   = backend_config.get("Options", {})
        else:
            self.backend_type = None

        self.precision = backend_options.get("Precision", None)

        # Define estimator based on backend type
        if self.backend_type is None:
            print(f"{getTimer()} INFO: No backend specified, state is going to evolve with direct matrix gates multiplication.")
            self.estimator = None
            self.sampler   = None
            return None, None

        elif self.backend_type == "StatevectorEstimator":
            print(f"{getTimer()} INFO: Using StatevectorEstimator (Ideal V2 Primitive).")
            self.estimator = StatevectorEstimator()
            self.sampler   = StatevectorSampler()
            return self.estimator, self.sampler
            
        elif self.backend_type == "Aer" or self.backend_type == "AerSimulator":
            
            print(f"{getTimer()} INFO: Using EstimatorV2 backed by AerSimulator with options {backend_options}")
            
            # Aer backend instance with options (e.g. shots, noise model, etc.) defined in the configuration. This backend will be used internally by the EstimatorV2.
            aer_backend = AerSimulator(**backend_options)
            
            # EstimatorV2 and SamplerV2 instance using the AerSimulator as backend.
            shots = backend_options.get("shots", 1024)
            self.estimator = EstimatorV2(backend=aer_backend, options={"default_shots": shots})
            self.sampler   = SamplerV2(backend=aer_backend, options={"default_shots": shots}) # Crea el sampler
            
            return self.estimator, self.sampler
        
        else:
            print(f"{getTimer()} WARNING: Backend type {self.backend_type} not implemented, reverting to direct matrix multiplication.")
            self.estimator = None
            self.sampler   = None
            return None, None

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
        else:
            print(f"{getTimer()} WARNING: Ansatz type {ansatz_type} not implemented...")
            return None
        
        ansatz_params = {
            "num_qubits": self.qubits_num,
            "entanglement": self.config["Ansatz"].get("Entanglement", "linear"),
            "reps": self.config["Ansatz"].get("Reps", 3),
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
            np.random.seed(42)
            # IMPROVED: Better parameter initialization (near identity instead of uniform random)
            init_strategy = self.config["Ansatz"].get("Init_Strategy", "random_small")
            if init_strategy == "random_small":
                # Start near identity: better convergence for VQE
                initial_state_params = np.random.normal(0, 0.1, self.ansatz.num_parameters)
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

        print(f"{getTimer()} INFO: Minimization config: method={minimize_params['method']}, maxiter={final_options['maxiter']}")
        print(f"{getTimer()} INFO: Initial parameters strategy: {init_strategy}")
        
        # IMPROVED: Track optimization history for diagnostics
        self.optimization_history = []
        iteration_count = [0]  # Use list to modify in nested function

        def callback(xk):
            energy = self.energy_cost_function(xk)
            self.optimization_history.append(energy)
            # Update progress bar
            try:    pbar.update(1)
            except: pass

        max_iter = final_options.get('maxiter', 2000)        
        with tqdm(total=max_iter, desc="VQE Optimization", unit="iter", file=sys.stdout, leave=True, dynamic_ncols=False) as pbar:
            result = minimize(self.energy_cost_function, initial_state_params, **minimize_params,
                            callback=callback)
            # Update bar to show actual iterations
            if hasattr(result, 'nit'):
                pbar.total = result.nit
                pbar.n = result.nit
                pbar.refresh()

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
        return calculateEnergy(
            ansatz_circuit, self.hamiltonian_prep,
            getattr(self, 'estimator', None),
            getattr(self, 'precision', None)
        )
    
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

         - observables_dataframe: DataFrame with time values of the evolution as index and columns the observables given in the list in the configuration
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
        else:
            print(f"{getTimer()} INFO: No Quench parameters, evolving with base hamiltonian.")
            self.hamiltonian_evol = self.hamiltonian_prep
        
        evolution_gate = gate(self.hamiltonian_evol, time=step, synthesis=synthesis(**synthesis_params))
        
        # Define initial state
        if self.estimator is None:
            state = Statevector.from_instruction(self.initial_state)
            initial_state = state.copy()
        else:
            state = self.initial_state.copy()
            initial_state = state.copy()
        
        # Define evolution method for null backend
        evolution_method = evolution_params.get("Evolution_Method", "MatrixExponential")

        # Prepare evolution data structures according to backend
        if evolution_method == "MatrixExponential":
            sparse_ham = self.hamiltonian_evol.to_matrix(sparse=True)
            state_data = initial_state.data.copy()

        observables        = evolution_params.get("Observables", {})
        observables_list   = observables.get("Observables_List", [])
        observables_params = observables.get("Observables_Params", {})
        observables_data   = {obs: [] for obs in observables_list}
        if "Pair_Creation" in observables_list:
            observables_data["Pair_Creation_electron"] = []
            observables_data["Pair_Creation_positron"] = []
            observables_data["Pair_Creation_balance"]  = []
            del observables_data["Pair_Creation"]

        # Iterate over time steps
        with tqdm(range(time_steps), desc="Evolving state", unit="step", file=sys.stdout, leave=True, dynamic_ncols=False) as pbar:
            for t in pbar:
                for obs in observables_list:
                    spec_params = observables_params.get(obs, None)
                    value = self.calculate_observable(obs, state, initial_state, spec_params=spec_params, estimator=self.estimator, precision=self.precision)
                    if obs == "Pair_Creation":
                        n_e, n_p = value
                        observables_data[f"{obs}_electron"].append(n_e)
                        observables_data[f"{obs}_positron"].append(n_p)
                        observables_data[f"{obs}_balance"].append(n_e - n_p)
                    else:
                        observables_data[obs].append(value)

                # Evolve state
                if self.estimator is None:
                    # Evolve state directly (no Aer backend)
                    if evolution_method == "MatrixExponential":
                        state_data = expm_multiply(-1j * sparse_ham * step, state_data)
                        state = Statevector(state_data)
                    else:
                        # Default: use gate evolution (slower but exact)
                        state = state.evolve(evolution_gate)
                else:
                    state.append(evolution_gate, range(self.qubits_num))

        time_array = np.linspace(0, total_time, time_steps)
        observables_dataframe = pd.DataFrame.from_records(observables_data, index=time_array)
        observables_dataframe.index.name = "Time"

        return state, observables_dataframe
    
    def calculate_observable(self, observable: str,
                             state: Statevector | QuantumCircuit,
                             initial_state: Statevector | None = None,
                             spec_params: Mapping | None = None,
                             estimator: BaseEstimatorV2 | None = None,
                             sampler: BaseSamplerV2 | None = None,
                             precision: float | None = None
                             ):
        '''
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
        else:
            print(f"{getTimer()} WARNING: Observable {observable} not implemented...")
            value = None
        
        return value
    

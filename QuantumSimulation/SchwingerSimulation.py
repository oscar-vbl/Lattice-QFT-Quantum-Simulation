"""
Main file for Schwinger simulation.
"""
from typing import Any, Mapping
# Importing types and classes from QuantumSimulation.core.types
from .core.types import QuantumState
from .core.types import HardwareConfig, NoiseErrorConfig, NoiseConfig
from .core.types import HamiltonianConfig, SchwingerParams
from .core.types import AnsatzConfig, InitialStateConfig
from .core.types import OptimizerConfig
from .core.types import SymmetryCheckConfig
from .core.types.dynamics import EvolutionConfig, EvolutionVariantConfig, QuenchConfig
from .core.types.mitigation import MitigationConfig, MitigationStepConfig
# Import PhysicsValidator for symmetry and state validation
from .core.loaders import load_symmetry_operators, load_validation_observables, load_evolution_observables
from .diagnostics import PhysicsValidator
# Import orchestration functions
from .hardware import setup_quantum_primitives
from .hamiltonians import build_hamiltonian
from .ansatz import build_parametrized_ansatz
from .dynamics import run_time_evolution, build_evolution_circuits
from .mitigation import apply_mitigation_pipeline
from .processing import extract_trajectory_dataframes
# Import algorithm solvers
from .algorithms import VQESolver

from dataclasses import replace
import numpy as np
from .circuits.transpilation import apply_transpilation
from .Utils import getTimer
from .observables import (
    EnergyObservable,
)

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
        initial_state: QuantumState | None = None,
    ):

        print("\n" + "#" * 70)
        print(f"{getTimer()} INFO: Initializing SchwingerSimulation class.")
        # Configuration parameters
        self.config = simulation_config
        # Number of qubits
        self.qubits_num = self.config["QubitsNumber"]
        # Initial state (if given, it will be used instead of optimizing for the vacuum state)
        self.initial_state = initial_state

        # Initialize hardware configuration and primitives
        self.hardware_config = self._parse_hardware_config()
        self.primitives = setup_quantum_primitives(self.hardware_config)

        # Parse hamiltonian configuration
        self.main_hamiltonian_config = self._parse_hamiltonian_config()

        # Parse ansatz configuration
        self.ansatz_config = self._parse_ansatz_config()

        # Parse diagnostics configuration
        self.diagnostics_config = self._parse_diagnostics_config()

        # Parse solver configuration
        self.optimizer_config = self._parse_optimizer_config()

        # Parse temporal evolution configuration
        self.evolution_config = self._parse_evolution_config()

    def run_simulation(
            self,
            initial_vqe_params: np.ndarray | None = None
        ):
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
        self.hamiltonian_prep = build_hamiltonian(self.main_hamiltonian_config)
        if self.hamiltonian_prep is None:
            raise ValueError("Failed to construct the Hamiltonian.")

        # Check symmetries
        # Load symmetry operators (e.g., TotalChargeOperator, GaussLawOperator)
        sym_operators = load_symmetry_operators(self.diagnostics_config["Symmetries"])
        
        # Initialize Validator with symmetry operators only
        self.validator = PhysicsValidator(operators_to_check=sym_operators)
        
        # Apply commutators and exact penalties, delegating math to the specified engine
        sym_checks_only = [config for config, _ in self.diagnostics_config["Symmetries"]]
        self.hamiltonian_prep = self.validator.enforce_symmetries(
            operator=self.hamiltonian_prep, 
            checks=sym_checks_only
        )

        # Get ansatz from config:
        # Initial config and type
        self.ansatz = build_parametrized_ansatz(
            self.qubits_num,
            self.ansatz_config,
            self.hamiltonian_prep
        )
        #self.ansatz = self.get_ansatz()

        # ISA Transpilation for primitives V2
        # Transpile if we use a real estimator or a simulator with backend
        if self.primitives.estimator is not None and self.primitives.backend is not None:
            print(f"{getTimer()} INFO: Transpiling ansatz to ISA circuit...")
            self.ansatz, layout_circuits = apply_transpilation(
                backend=self.primitives.backend,
                circuit=self.ansatz,
                layout_circuits=[self.hamiltonian_prep] + sym_operators
            )
            self.hamiltonian_prep = layout_circuits[0]
            sym_operators = layout_circuits[1:]

        # Get parameters that minimize initial ansatz energy
        if self.initial_state is None:
            #self.initial_state, self.vacuum_energy, self.vacuum_parameters = (
            #    self.get_vacuum()
            #)
            self.energy_observable = EnergyObservable(
                operator=self.hamiltonian_prep,
                precision=self.primitives.precision,
            )

            vqe_solver = VQESolver(
                ansatz=self.ansatz,
                minimize_observable=self.energy_observable,
                hamiltonian=self.hamiltonian_prep,
                config=self.optimizer_config, # Need definition
                primitives=self.primitives
            )
            self.vqe_result = vqe_solver.solve(initial_params=initial_vqe_params)
            self.initial_state, self.vacuum_energy, self.vacuum_parameters = (
                self.vqe_result.state, self.vqe_result.energy, self.vqe_result.parameters
            )
        else:
            if self.primitives.estimator is not None and self.primitives.backend is not None:
                print(f"{getTimer()} INFO: Transpiling initial state to ISA circuit...")
                self.initial_state, layout_circuits = apply_transpilation(
                    backend=self.primitives.backend,
                    circuit=self.initial_state,
                    layout_circuits=[self.hamiltonian_prep] + sym_operators
                )
                self.hamiltonian_prep = layout_circuits[0]
                sym_operators = layout_circuits[1:]

            self.get_vacuum_energy_from_initial()

            self.vacuum_parameters = None

        print(f"{getTimer()} INFO: Initial energy = {self.vacuum_energy}")

        # Checks on initial state
        val_observables = load_validation_observables(
            configs_with_params=self.diagnostics_config["StateValidation"],
            reference_state=self.initial_state,
            precision=self.primitives.precision,
            shots=getattr(self.primitives, "shots", None)
        )
        
        # Append the new observables to the existing validator registry
        self.validator.operators_to_check.extend(val_observables)
        self.validator.operator_names = [op.name for op in self.validator.operators_to_check]
        
        # Validate the prepared state
        val_checks_only = [config for config, _ in self.diagnostics_config["StateValidation"]]
        self.ground_state_diagnostics = self.validator.validate_state(
            state=self.initial_state, 
            checks=val_checks_only
        )
        self.ground_state_diagnostics["vacuum_energy"] = self.vacuum_energy
        # Store diagnostics for later inspection
        #self.ground_state_diagnostics = {
        #    "vacuum_energy": self.vacuum_energy,
        #    "vacuum_charge": Q_total_inicial
        #    if self.config["Ansatz"].get("Ensure_Zero_Charge", False)
        #    else None,
        #    "optimization_history": self.optimization_history
        #    if hasattr(self, "optimization_history")
        #    else [],
        #}

        #######################
        # 2. Temporal evolution
        #######################
        self._execute_dynamics()
        #if self.config.get("Temporal Evolution", {}).get("Active", False):
        #    (
        #        self.final_state,
        #        self.evolution_data,
        #        self.evolution_error,
        #        self.all_trotter_evolution_data,
        #        self.all_trotter_evolution_error,
        #    ) = self.evolve_state()

        # Additional results and calculations (not developed in this module)
        # Check Results for results examples
        print(f"{getTimer()} INFO: Simulation ended.")
        print("#" * 70 + "\n")

    def get_vacuum_energy_from_initial(self):
        """
        Calculate vacuum energy from given initial_state
        """
        energy_obs = EnergyObservable(self.hamiltonian_prep, self.primitives.precision)
        if self.primitives.estimator is not None:
            pub = energy_obs.get_pub(self.initial_state)
            result = self.primitives.estimator.run([pub]).result()
            self.vacuum_energy = energy_obs.process_pub_result(result[0])[0]
        else:
            self.vacuum_energy = energy_obs.calculate_exact(self.initial_state)[0]


    def _parse_hardware_config(self) -> HardwareConfig:
        """
        Parses the raw JSON backend configuration into a typed HardwareConfig object.
        """
        # Extract backend configuration from the simulation config
        raw_hardware_config = self.config.get("Backend", {})

        # Check if noise model is provided in the backend configuration
        raw_noise = raw_hardware_config.get("Noise Model")
        if raw_noise and "Errors" in raw_noise:
            error_list = [
                NoiseErrorConfig(
                    error_type=e.get("Type"),
                    num_qubits=e.get("Num_Qubits", 1),
                    probability=e.get("Probability", 0.01),
                    gates=e.get("Gates", [])
                ) for e in raw_noise["Errors"]
            ]
            noise_config = NoiseConfig(errors=error_list)
        else:
            noise_config = None
        
        # Build hardware configuration object
        return HardwareConfig(
            backend_type=raw_hardware_config.get("Type"),
            shots=raw_hardware_config.get("Shots"),
            precision=raw_hardware_config.get("Precision"),
            hardware_name=raw_hardware_config.get("Hardware_Name", "least_busy"),
            noise_model_config=noise_config,
            options=raw_hardware_config.get("Options", {})
        )

    def _parse_hamiltonian_config(self) -> HamiltonianConfig:
        """Parses raw JSON into a typed HamiltonianConfig object."""

        raw_hamiltonian_config = self.config.get("Hamiltonian", {})

        model_type = raw_hamiltonian_config.get("Type", "Schwinger")
        gauge      = raw_hamiltonian_config.get("Gauge", "Temporal")
        raw_params = raw_hamiltonian_config.get("Parameters", {})
        
        if model_type == "Schwinger":
            parsed_params = SchwingerParams(
                L=raw_params["L"],
                a=raw_params["a"],
                m=raw_params["m"],
                e0=raw_params.get("e0", 0.0),
                periodic=raw_params.get("periodic", False)
            )
        else:
            parsed_params = raw_params  # Fallback for future models
            
        return HamiltonianConfig(
            model_type=model_type,
            gauge=gauge,
            parameters=parsed_params
        )

    def _parse_ansatz_config(self) -> AnsatzConfig:
        """Parses raw JSON into a typed AnsatzConfig object."""

        raw_ansatz = self.config.get("Ansatz", {})

        ansatz_type=raw_ansatz.get("Type", "EfficientSU2")
        entanglement=raw_ansatz.get("Entanglement", "linear")
        reps=raw_ansatz.get("Reps", 3)
        additional_params=raw_ansatz.get("AdditionalParams", {})

        raw_initial = raw_ansatz.get("Initial State", {})
        
        initial_config = InitialStateConfig(
            vacuum=raw_initial.get("Vacuum", True),
            staggered=raw_initial.get("Staggered", True)
        )
        
        return AnsatzConfig(
            ansatz_type=ansatz_type,
            initial_state=initial_config,
            entanglement=entanglement,
            reps=reps,
            additional_params=additional_params
        )

    def _parse_diagnostics_config(self) -> dict[str, list[SymmetryCheckConfig]]:
        """
        Parses the JSON configuration into typed diagnostic contracts.
        Separates checks for Hamiltonian Symmetries and State Validation.
        """
        
        parsed_configs = {
            "Symmetries": [],
            "StateValidation": []
        }
        
        diag_config = self.config.get("Diagnostics", {})
        
        # 1. Parse Hamiltonian Symmetry Checks
        for sym_json in diag_config.get("Symmetries", []):
            parsed_configs["Symmetries"].append(
                (
                    SymmetryCheckConfig(
                        name=sym_json.get("Check_Operator"),
                        target_value=sym_json.get("Target_Value", 0.0),
                        tolerance=sym_json.get("Tolerance", 0.01),
                        penalty_weight=sym_json.get("Penalty_Weight", None)
                    ),
                    self.main_hamiltonian_config.parameters
                )
            )
            
        # 2. Parse State Validation Checks
        for state_json in diag_config.get("StateValidation", []):
            parsed_configs["StateValidation"].append(
                (
                    SymmetryCheckConfig(
                        name=state_json.get("Check_Observable"),
                        target_value=state_json.get("Target_Value", 0.0),
                        tolerance=state_json.get("Tolerance", 0.01)
                    ),
                    self.main_hamiltonian_config.parameters
                )
            )
            
        # 3. Backward compatibility for older JSONs
        legacy_lambda = self.config.get("Hamiltonian", {}).get("Lambda_Charge_Penalty", 0)
        if legacy_lambda and not parsed_configs["Symmetries"]:
            parsed_configs["Symmetries"].append(
                (
                    SymmetryCheckConfig(
                        name="Total_Charge_Operator",
                        target_value=0.0,
                        tolerance=0.01,
                        penalty_weight=legacy_lambda
                    ),
                    self.main_hamiltonian_config.parameters
                )
            )
        legacy_ensure_charge = self.config.get("Ansatz", {}).get("Ensure_Zero_Charge", False)
        if legacy_ensure_charge:
            parsed_configs["StateValidation"].append(
                (
                    SymmetryCheckConfig(
                        name="Total_Charge",
                        target_value=0.0,
                        tolerance=0.01,
                    ),
                    self.main_hamiltonian_config.parameters,
                )
            )            
            
        return parsed_configs

    def _parse_optimizer_config(self) -> OptimizerConfig:
        """Parses raw JSON into a typed OptimizerConfig object."""
        raw_ansatz = self.config.get("Ansatz", {})
        raw_minimizer = raw_ansatz.get("Minimizer", {})
        raw_options = raw_minimizer.get("Options", {})

        return OptimizerConfig(
            method=raw_minimizer.get("Method", "COBYLA"),
            maxiter=raw_options.get("maxiter", 2000),
            tol=raw_options.get(
                "tol",
                raw_options.get("ftol", 1e-6),
            ),
            use_gradient=raw_ansatz.get(
                "Use Gradient Cost",
                False,
            ),
            init_strategy=raw_ansatz.get(
                "Init_Strategy",
                "random_small",
            ),
            init_max=raw_ansatz.get("Init_Max", 0.1),
            options={
                key: value
                for key, value in raw_options.items()
                if key not in {"maxiter", "tol"}
            },
        )

    def _parse_evolution_config(self) -> EvolutionConfig:
        """
        Parses the raw JSON temporal evolution configuration into strictly typed domain models.
        Maps legacy configuration keys to the new variant and mitigation pipeline structures.
        """

        raw_evol = self.config.get("Temporal Evolution", {})
        
        # ==========================================
        # 1. Parse Quench Configuration
        # ==========================================
        raw_quench = raw_evol.get("Quench", {})
        quench_config = QuenchConfig(
            active=raw_quench.get("Active", False),
            parameter_overrides=raw_quench.get("Parameters_to_Change", {})
        )
        
        # ==========================================
        # 2. Parse Algorithmic Mitigation Pipeline
        # ==========================================
        raw_mitigation = raw_evol.get("Algorithmic_Mitigation", {})
        mitigation_enabled = raw_mitigation.get("Apply", False)
        
        mitigation_steps = []
        if mitigation_enabled:
            # Check if explicit steps are provided in newer JSONs, otherwise build a legacy default
            raw_steps = raw_mitigation.get("Steps", [])
            if not raw_steps:
                mitigation_steps.append(
                    MitigationStepConfig(
                        method="richardson",
                        options=raw_mitigation.get("Options", {})
                    )
                )
            else:
                for step in raw_steps:
                    mitigation_steps.append(
                        MitigationStepConfig(
                            method=step.get("Method", "richardson"),
                            options=step.get("Options", {})
                        )
                    )
                    
        mitigation_config = MitigationConfig(
            enabled=mitigation_enabled,
            steps=tuple(mitigation_steps),
            preferred_output=raw_mitigation.get("Preferred_Output", None)
        )
        
        # ==========================================
        # 3. Parse Trajectory Variants (Trotter Steps)
        # ==========================================
        # Fallback to single "dt" variant if not specified
        raw_variants = raw_evol.get("Trotter_Steps", {"dt": {"step_multiplier": 1.0}})
        
        # Global fallback for legacy 'Evolution_Method' (e.g., MatrixExponential)
        global_fallback_method = raw_evol.get("Evolution_Method", "trotter")
        if global_fallback_method == "MatrixExponential":
            global_fallback_method = "matrix_exponential"
            
        variants = []
        for key, v_conf in raw_variants.items():
            variants.append(
                EvolutionVariantConfig(
                    key=key,
                    method=v_conf.get("Method", global_fallback_method).lower(),
                    step_multiplier=v_conf.get("step_multiplier", 1.0),
                    options=v_conf.get("Options", {})
                )
            )

        # ==========================================
        # 4. Parse Observables
        # ==========================================
        raw_obs = raw_evol.get("Observables", {})
        observables_list = tuple(raw_obs.get("Observables_List", []))
        observable_options = raw_obs.get("Observables_Params", {})

        # ==========================================
        # 5. Parse Gate & Synthesis Configuration
        # ==========================================
        raw_gate = raw_evol.get("Evolution_Gate", {})
        raw_synthesis = raw_gate.get("Synthesis", "suzuki_trotter").lower()
        if raw_synthesis == "trottersuzuki":
            raw_synthesis = "suzuki_trotter"

        # ==========================================
        # 6. Assemble Full Evolution Domain Model
        # ==========================================
        return EvolutionConfig(
            active=raw_evol.get("Active", False),
            total_time=raw_evol.get("Total_Time", 1.0),
            time_steps=raw_evol.get("Time_Steps", 10),
            gate_type=raw_gate.get("Type", "pauli").lower(),
            synthesis=raw_synthesis,
            synthesis_options=raw_gate.get("Synthesis_Params", {}),
            variants=tuple(variants),
            preferred_variant=raw_evol.get("Preferential_Trotter", None),
            quench=quench_config,
            observables=observables_list,
            observable_options=observable_options,
            mitigation=mitigation_config if mitigation_enabled else None,
            store_states=raw_evol.get("Store_States", False)
        )


    def _execute_dynamics(self) -> None:
        """
        Executes the temporal evolution pipeline, calculates observables,
        applies generic error mitigation, and generates final DataFrames.
        """
        # 1. Parse typed configuration        
        if not self.evolution_config.active:
            return

        # 2. Apply physical quench if required
        if self.evolution_config.quench.active:
            self.hamiltonian_evol = build_hamiltonian(
                self.main_hamiltonian_config, 
                override_params=self.evolution_config.quench.parameter_overrides
            )
            evol_params = replace(
                self.main_hamiltonian_config.parameters, 
                **self.evolution_config.quench.parameter_overrides
            )
        else:
            self.hamiltonian_evol = self.hamiltonian_prep
            evol_params = self.main_hamiltonian_config.parameters

        active_observables = load_evolution_observables(
            observables_list=self.evolution_config.observables,
            observable_options=self.evolution_config.observable_options,
            params=evol_params,
            hamiltonian=self.hamiltonian_evol,
            reference_state=self.initial_state,
            precision=self.primitives.precision,
            shots=getattr(self.primitives, "shots", None)
        )

        # 3. Build circuits
        evolution_circuits = build_evolution_circuits(
            hamiltonian=self.hamiltonian_evol,
            config=self.evolution_config,
            primitives=self.primitives,
            num_qubits=self.qubits_num
        )

        # 4. Execute Time Engine (Returns typed EvolutionResult)
        evolution_result = run_time_evolution(
            initial_state=self.initial_state,
            hamiltonian=self.hamiltonian_evol,
            observables_objects=active_observables,
            config=self.evolution_config,
            primitives=self.primitives,
            num_qubits=self.qubits_num,
            evolution_circuits=evolution_circuits
        )

        # 5. Apply Error Mitigation Pipeline (Generic)
        if self.evolution_config.mitigation and self.evolution_config.mitigation.enabled:
            evolution_result = apply_mitigation_pipeline(
                evolution_result, self.evolution_config.mitigation
            )

        # 6. Postprocessing to Pandas DataFrames
        
        self.evolution_result = evolution_result
        self.final_state = evolution_result.final_state  # Safely resolves via state_source_key
        
        # Expose the preferred trajectory as the main DataFrames for the user
        preferred_trajectory = evolution_result.preferred
        self.observables_dataframe, self.observables_dataframe_error = extract_trajectory_dataframes(
            preferred_trajectory
        )

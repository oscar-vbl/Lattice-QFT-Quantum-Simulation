# Main file for Schwinger simulation

import numpy as np
import pandas as pd
import os
from typing import Callable, Any
from qiskit.circuit.quantumcircuit import QuantumCircuit
from circuitBuilder import buildCircuit, addGate
from Operators import buildSchwingerHamiltonianTemporalGauge, buildChargeOperatorMinimal, gauss_operator
from Plots import simplePlot
from Utils import sortEigenstates, drawCircuitLatex, getTimer, saveJsonConfig, loadJsonConfig, \
getValidFileName
from Utils import PLOTS_FOLDER as plt_folder
from Calculations import calculateEnergy, calculateVaccumPersistance, calculateGaussLawViolation, checkChargeSymmetry
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit.library import efficient_su2, n_local, excitation_preserving
from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
#from qiskit.circuit.library import EfficientSU2 # Old qiskit version
from scipy.optimize import minimize

class SchwingerSimulation:
    def __init__(self, simulation_config):
        self.config = simulation_config
        self.qubits_num = self.config["QubitsNumber"]

    def start_simulation(self):
        ####################
        # 1. Get main params for the simulation
        ####################
        # Get hamiltonian
        self.hamiltonian = self.get_hamiltonian()
        if self.hamiltonian is None: return

        # Check symmetries
        e0 = self.config["Hamiltonian"]["Parameters"].get("e0", 0)
        hasSym, Q_op = checkChargeSymmetry(self.hamiltonian, e0=e0)
        if not hasSym: return

        # Add charge penalty in order to minimize charge symmetry violations
        lambda_penalty = self.config["Hamiltonian"].get("Lambda_Charge_Penalty", 0)
        if lambda_penalty > 0:
            print(f"{getTimer()} INFO: Adding charge penalty term with lambda = {lambda_penalty}")
            self.hamiltonian = (self.hamiltonian + lambda_penalty * Q_op.dot(Q_op)).simplify()

        # Get ansatz from config:
        # Initial config and type
        self.ansatz = self.get_ansatz()

        # Get parameters that minimize initial ansatz energy
        self.initial_state, self.vaccum_energy = self.get_vaccum()
        print(f"{getTimer()} INFO: Initial energy = {self.vaccum_energy}")
        if self.config["Ansatz"].get("Ensure_Zero_Charge", False):
            Q_total_inicial = Statevector(self.initial_state).expectation_value(Q_op).real
            print(f"{getTimer()} INFO: Initial charge = {Q_total_inicial}")
            if abs(Q_total_inicial) > 0.01:
                print(f"{getTimer()} WARNING: Initial state does not have zero charge, review ansatz and minimization parameters.")
                return
        
        ####################
        # 2. Temporal evolution
        ####################
        self.final_state, self.evolution_data = self.evolve_state()

        # Plot data (not developed yet, at very first stage)
        # Check ResultsSimulation for simple results examples


    def get_hamiltonian(self):
        ham_type   = self.config["Hamiltonian"]["Type"]
        ham_params = self.config["Hamiltonian"]["Parameters"]
        ham_gauge  = self.config["Hamiltonian"].get("Gauge", "Temporal")
        if ham_type == "Schwinger":
            if ham_gauge == "Temporal":
                return func_return(buildSchwingerHamiltonianTemporalGauge, ham_params)
            else:
                print(f"{getTimer()} WARNING: Gauge {ham_gauge} not implemented...")
                return None
        else:
            print(f"{getTimer()} WARNING: Hamiltonian type {ham_type} not implemented...")
            return None

    def get_ansatz(self):
        ansatz_type     = self.config["Ansatz"]["Type"]
        in_state_params = self.config["Ansatz"]["Initial State"]

        if in_state_params.get("Vaccum", True):
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
            print(f"{getTimer()} WARNING: Ansatz state for not vaccum state not implemented...")
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
            #"initial_state": initial_circuit, # Old qiskit version
            **self.config["Ansatz"].get("AdditionalParams", {})
        }
        ansatz = func_return(ansatz_func, ansatz_params)
        assert isinstance(ansatz, QuantumCircuit), f"{getTimer()} WARNING: Ansatz function did not return a QuantumCircuit, review ansatz parameters..."
        if ansatz is not None:
            # Evolve from initial vaccum
            ansatz.compose(initial_circuit, inplace=True)
        return ansatz

    def get_vaccum(self):
        initial_state_params = self.config["Ansatz"].get("Initial Parameters", None)
        if not initial_state_params:
            initial_state_params = np.random.random(self.ansatz.num_parameters) * 2 * np.pi

        if not self.config["Ansatz"].get("Minimizer", None):
            self.config["Ansatz"]["Minimizer"] = {}

        minimize_params = {
            "method": self.config["Ansatz"]["Minimizer"].get("Method", "COBYLA"),
            "options": self.config["Ansatz"]["Minimizer"].get("Options", {"maxiter": 100}),
            **self.config["Ansatz"]["Minimizer"].get("AdditionalParams", {})
        }

        result = minimize(self.energy_cost_function, initial_state_params, **minimize_params)
        vaccum_state  = self.ansatz.assign_parameters(result.x)
        vaccum_energy = result.fun
        return vaccum_state, vaccum_energy

    def energy_cost_function(self, params):
        '''
        Cost funcion to minimize energy of the statevector
        obtained by the ansatz with given parameters,
        given self.hamiltonian and self.ansatz.
        '''
        state = Statevector(self.ansatz.assign_parameters(params))
        return state.expectation_value(self.hamiltonian).real
    
    def evolve_state(self):
        '''
        Evolves the initial state with the configuration "Temporal Evolution" at self.config

        Returns: 
         - state: Statevector of the final state of the evolution.

         - observables_dataframe: DataFrame with time values of the evolution as index and columns the observables given in the list in the configuration
        '''
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

        evolution_gate = gate(self.hamiltonian, time=step, synthesis=synthesis(**synthesis_params))
        
        #state = Statevector(self.initial_state)
        #initial_state = Statevector(self.initial_state)
        state = Statevector.from_instruction(self.initial_state)
        initial_state = state.copy()

        observables        = evolution_params.get("Observables", {})
        observables_list   = observables.get("Observables_List", [])
        observables_params = observables.get("Observables_Params", {})
        observables_data   = {obs: [] for obs in observables_list}
        for t in range(time_steps):
            for obs in observables_list:
                spec_params = observables_params.get(obs, None)
                value = self.calculate_observable(obs, state, initial_state, spec_params=spec_params)
                observables_data[obs].append(value)
            state = state.evolve(evolution_gate)

        observables_dataframe = pd.DataFrame.from_records(observables_data, index=np.linspace(0,total_time,time_steps))
        return state, observables_dataframe
    
    def calculate_observable(self, observable,
                             state: Statevector,
                             initial_state: Statevector | None = None,
                             spec_params: dict | None = None,
                             ):
        '''Calculate the expectation value of a given observable.'''
        
        if observable == "Energy":
            value = calculateEnergy(state, self.hamiltonian)
        elif observable == "Persistance":
            value = calculateVaccumPersistance(state, initial_state)
        elif observable == "Gauss_Law_Violation":
            value = calculateGaussLawViolation(state, self.qubits_num)
        else:
            print(f"{getTimer()} WARNING: Observable {observable} not implemented...")
            value = None
        
        return value
    
def func_return(func: Callable, params: dict = {}, default=None, expect_type: type | None = None) -> Any:
    '''try return a function given its parameters, else return default'''
    try:
        result = func(**params)
        if expect_type is not None:
            assert isinstance(result, expect_type), f"{getTimer()} WARNING: Function {func.__name__} did not return expected type {expect_type}, got {type(result)}. Review parameters..."
        return result
    except Exception as e:
        print(f"{getTimer()} WARNING: Exception {e} raised, review parameters...")
        return default


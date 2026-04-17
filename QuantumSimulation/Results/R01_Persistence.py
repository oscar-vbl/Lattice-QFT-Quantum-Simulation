import sys
import numpy as np
import pandas as pd
import copy
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())
from _config import PLOTS_FOLDER as plt_folder
from _config import DATA_FOLDER  as data_folder
from ResultsAnalysis import fit_persistence
from Utils import getTimer, loadJsonConfig, parseDictToPlot, load_data, save_data
from Plots import plot_gamma_vs_qubitNum, plot_persistenece_vs_time_regimes, plot_gamma_vs_e0, plot_gamma_vs_electricField
from R00_ResultsCommon import get_simulation_data, load_evolution_and_initial

def fit_persistence_e0(config,
                       e0_values,
                       e0_data=None,
                       initial_state=None):
    qubits_num = config["QubitsNumber"]

    fit_results = {e0: {
            "Gamma_Simulated": None,
            "Gamma_Analytical": None,
            "E_Physical": None,
        } for e0 in e0_values
    }

    for e0 in e0_values:

        if e0_data is not None:
            try:
                evolution_data = e0_data[e0]["evolution_data"]
                initial_state  = e0_data[e0]["initial_state"]
            except KeyError:
                print(f"{getTimer()} WARNING: Data for e0={e0} not found, running simulation.")
                simulator, duration = get_simulation_data(config, initial_state=initial_state)
                evolution_data = simulator.evolution_data
                initial_state  = simulator.initial_state
                print(f"{getTimer()} INFO: Simulation for {qubits_num} qubits and e0={e0} took {(duration):.2f} seconds.")
        else:
            simulator, duration = get_simulation_data(config, initial_state=initial_state)
            evolution_data = simulator.evolution_data
            initial_state  = simulator.initial_state
            print(f"{getTimer()} INFO: Simulation for {qubits_num} qubits and e0={e0} took {(duration):.2f} seconds.")

        e0_config = copy.deepcopy(config)
        e0_config["Temporal Evolution"]["Quench"]["Parameters_to_Change"]["e0"] = e0

        simulated_gamma, gamma_analytical, gamma_err, eE_evol, cut_off_times = fit_persistence(
            evolution_data, e0_config, initial_state,
            use_offset=False,
            print_info=False
        )

        fit_results[e0]["Gamma_Simulated"]  = simulated_gamma
        fit_results[e0]["Gamma_Analytical"] = gamma_analytical
        fit_results[e0]["E_Physical"]       = eE_evol

    fit_results_df = pd.DataFrame.from_dict(fit_results, orient="index")
    fit_results_df.index.name = "e0"

    return fit_results_df

def fit_persistence_qubits_num(config,
                               qubits_num_values,
                               qubits_num_data=None):

    fit_results = {qubits_num: {
            "Gamma_Simulated": None,
            "Gamma_Analytical": None,
            "E_Physical": None,
            "Cut_Off_Times": None
        } for qubits_num in qubits_num_values
    }
    
    for qubits_num in qubits_num_values:
        if qubits_num_data is not None:
            try:
                evolution_data = qubits_num_data[qubits_num]["evolution_data"]
                initial_state  = qubits_num_data[qubits_num]["initial_state"]
            except KeyError:
                print(f"{getTimer()} WARNING: Data for L={qubits_num} not found, running simulation.")
                simulator, duration = get_simulation_data(config)
                evolution_data = simulator.evolution_data
                initial_state  = simulator.initial_state
                print(f"{getTimer()} INFO: Simulation for {qubits_num} qubits took {(duration):.2f} seconds.")
        else:
            simulator, duration = get_simulation_data(config)
            evolution_data = simulator.evolution_data
            initial_state  = simulator.initial_state
            print(f"{getTimer()} INFO: Simulation for {qubits_num} qubits took {(duration):.2f} seconds.")

        qubits_num_config = copy.deepcopy(config)
        qubits_num_config["QubitsNumber"] = qubits_num
        qubits_num_config["Hamiltonian"]["Parameters"]["L"] = qubits_num
        

        simulated_gamma, gamma_analytical, gamma_err, eE_evol, cut_off_times = fit_persistence(
            evolution_data, qubits_num_config, initial_state,
            use_offset=False,
            print_info=False
        )

        fit_results[qubits_num]["Gamma_Simulated"]  = simulated_gamma
        fit_results[qubits_num]["Gamma_Analytical"] = gamma_analytical
        fit_results[qubits_num]["E_Physical"]       = eE_evol
        fit_results[qubits_num]["Cut_Off_Times"]    = cut_off_times

    fit_results_df = pd.DataFrame.from_dict(fit_results, orient="index")
    fit_results_df.index.name = "Qubits_Num"

    return fit_results_df


if __name__ == "__main__":
    # Load Config
    config = loadJsonConfig("SchwingerSimulation_Persistence.json")

    # Set to False if you want to run all simulations from scratch
    # True to load previously simulated data (if available) or run simulation if not found.
    USE_SIMULATED_DATA = True

    if True: # 1. Get evolution data for different values of L
        analysis_name = f"decay_rate_qubits_num"
        file_name     = f"decay_rate_qubits_num"

        # Get values of L
        min_qubit_num, max_qubit_num = 10, 20
        step = 2
        qubit_num_values = np.arange(min_qubit_num, max_qubit_num+step, step)

        # Get needed data for each L
        analysis_name = "qubits_num_quench"
        qubits_num_data = load_evolution_and_initial(analysis_name, qubit_num_values,
                                                 evolution_temp="qubits_num_{value}_quench_data.csv",
                                                 initial_state_temp="qubits_num_{value}_initial_state.qpy",
                                                 backup_config=config,
                                                 backup_key="L",
                                                 use_simulated_data=USE_SIMULATED_DATA)

        # Get results
        fit_results_df = fit_persistence_qubits_num(config, qubit_num_values, qubits_num_data)
        plot_params = parseDictToPlot(
            {**config["Hamiltonian"]["Parameters"], **config["Temporal Evolution"]["Quench"]["Parameters_to_Change"]},
            remove_keys=["L"],
            rename_keys={"e0": "$\\varepsilon_0$"})
        fig, ax = plot_gamma_vs_qubitNum(fit_results_df, params=plot_params)
        # Save plot and data
        save_data(fig,            analysis_name, f"{file_name}.png", rootPath=plt_folder, dpi=300)
        save_data(fit_results_df, analysis_name, f"{file_name}.csv", rootPath=plt_folder)

        # Calculate deviation to find best fit
        fit_results_df["Deviation"] = np.abs(1 - fit_results_df["Gamma_Analytical"]/ fit_results_df["Gamma_Simulated"])
        best_qubit_num = int(fit_results_df["Deviation"].idxmin())

    if True: # 2. Show best fit
        analysis_name = f"best_{best_qubit_num}_decay_rate"
        file_name     = f"best_{best_qubit_num}_decay_rate"

        # Get evolution data and initial state for best simulation
        evolution_data = qubits_num_data[best_qubit_num]["evolution_data"]
        initial_state  = qubits_num_data[best_qubit_num]["initial_state"]
        # Use config from best simulation
        qubits_num_config = copy.deepcopy(config)
        qubits_num_config["QubitsNumber"] = best_qubit_num
        qubits_num_config["Hamiltonian"]["Parameters"]["L"] = best_qubit_num

        # Fit persistence to get plot showing exponential fit
        fig, axes = fit_persistence(
            evolution_data, qubits_num_config, initial_state,
            use_offset=False,
            return_plot=True
        )
        # Save plot and data
        save_data(fig,            analysis_name, f"{file_name}.png", rootPath=plt_folder, dpi=300)
        save_data(fit_results_df, analysis_name, f"{file_name}.csv", rootPath=plt_folder)

    if True: # 3. Plot persistence and regimes for best fit
        analysis_name = f"best_{best_qubit_num}_persistenece_vs_time"
        file_name     = f"best_{best_qubit_num}_persistenece_vs_time"
        # Use evoltion data and cut off times to plot persistence vs time with different regimes
        cut_off_times = fit_results_df.loc[best_qubit_num, "Cut_Off_Times"]
        plot_params = parseDictToPlot(
            {**qubits_num_config["Hamiltonian"]["Parameters"], **qubits_num_config["Temporal Evolution"]["Quench"]["Parameters_to_Change"]},
            remove_keys=[],
            rename_keys={"e0": "$\\varepsilon_0$"})
        fig, ax = plot_persistenece_vs_time_regimes(evolution_data, cut_off_times, plot_params)
        # Save plot and data
        save_data(fig,            analysis_name, f"{file_name}.png", rootPath=plt_folder, dpi=300)
        save_data(fit_results_df, analysis_name, f"{file_name}.csv", rootPath=plt_folder)

    if True: # 4. Show best fit for different values of e0
        # Take best simulation
        qubits_num = best_qubit_num

        analysis_name = f"e0_quench_{qubits_num}qubits"
        file_name     = f"e0_quench_{qubits_num}qubits"

        # Define e0 values to show Gamma
        min_e0, max_e0 = 0.4, 0.98
        step = 0.02
        e0_values = np.arange(min_e0, max_e0+step, step)

        min_e0_2, max_e0_2 = 1.0, 1.5
        step2 = 0.1
        e0_values = np.concatenate((e0_values, np.arange(min_e0_2, max_e0_2+step2, step2)))

        # Initial state for simulation (in case data is not found)
        # Can get from previous simulation, it's the same for all e0 since e0 is quenched
        try:    initial_state = qubits_num_data[qubits_num]["initial_state"]
        except: initial_state = None
        
        e0_data = load_evolution_and_initial(analysis_name, e0_values,
                                        evolution_temp="e0_{value}_quench_data.csv",
                                        initial_state_temp="e0_{value}_initial_state.qpy",
                                        backup_config=qubits_num_config,
                                        backup_initial_state=initial_state,
                                        backup_key="e0",
                                        backup_key_is_quench=True,
                                        use_simulated_data=USE_SIMULATED_DATA)

        # Get results of gamma(e0)
        fit_results_df = fit_persistence_e0(qubits_num_config, e0_values, e0_data)

        # Plot gamma(e0)
        plot_params = parseDictToPlot(
            qubits_num_config["Hamiltonian"]["Parameters"],
            remove_keys=["e0"],
            rename_keys={})
        fig, ax = plot_gamma_vs_e0(fit_results_df, params=plot_params)
        # Save plot and data
        save_data(fig,            analysis_name, f"{file_name}.png", rootPath=plt_folder, dpi=300)
        save_data(fit_results_df, analysis_name, f"{file_name}.csv", rootPath=plt_folder)

    if True: # 5. Plot log Gamma vs 1/e0 in logartihmic scale for Schwinger regime
        analysis_name = f"best_{qubits_num}_logPersistenece_vs_electricField"
        file_name     = f"best_{qubits_num}_logPersistenece_vs_electricField"
        min_e0, max_e0 = 0.4, 0.541 # Found by inspection

        # Take e0 values from previous fit results
        e0_values = fit_results_df.index.values
        e0_values = e0_values[(e0_values >= min_e0) & (e0_values <= max_e0)]
        fit_reg_results_df = fit_results_df.loc[e0_values]

        # Values of gamma
        gamma_simulated = fit_reg_results_df["Gamma_Simulated"].values
        # Values of physical field
        field_values = fit_reg_results_df["E_Physical"].apply(lambda x: np.mean(x)).values

        # Plot log Gamma/e0 vs 1/e0 for Schwinger regime
        plot_params = parseDictToPlot(
            qubits_num_config["Hamiltonian"]["Parameters"],
            remove_keys=["e0"],
            rename_keys={})
        fig, axes, fit_params = plot_gamma_vs_electricField(gamma_simulated, e0_values, field_values, plot_params)
        # Save plot and data
        save_data(fig,            analysis_name, f"{file_name}.png", rootPath=plt_folder, dpi=300)
        save_data(fit_results_df, analysis_name, f"{file_name}.csv", rootPath=plt_folder)





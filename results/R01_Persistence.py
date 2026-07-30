"""
R01 - Schwinger Effect Persistence Analysis (v3)
=================================================
Refactored to load SchwingerSimulation .pkl instances via load_simulator_instance
instead of separate evolution_data / initial_state files.

Each .pkl contains a fully serialized SchwingerSimulation object with:
    simulator.evolution_data              — main observables DataFrame (index=Time)
    simulator.evolution_error             — per-point statistical error DataFrame
    simulator.all_trotter_evolution_data  — dict {trotter_key: DataFrame}
    simulator.all_trotter_evolution_error — dict {trotter_key: DataFrame}
    simulator.initial_state               — QuantumCircuit of the vacuum state
    simulator.config                      — full simulation config

The Trotter error is estimated by comparing two Trotter configurations stored
in all_trotter_evolution_data (keys defined in config["Temporal Evolution"]["Trotter_Steps"]).
"""

import sys
import numpy as np
import pandas as pd
import copy
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())
from QuantumSimulation._config import PLOTS_FOLDER as plt_folder
from QuantumSimulation.ResultsAnalysis import fit_persistence
from QuantumSimulation.Utils import getTimer, loadJsonConfig, parseDictToPlot, save_data
from QuantumSimulation.Plots import (
    plot_gamma_vs_qubitNum,
    plot_persistenece_vs_time_regimes,
    plot_gamma_vs_e0,
    plot_gamma_vs_electricField,
)
from R00_ResultsCommon import get_simulation_data, load_simulator_instance, load_evolution_and_initial


# ──────────────────────────────────────────────────────────────────────────── #
#  Helper: extract data from simulator instance                                #
# ──────────────────────────────────────────────────────────────────────────── #

def _get_simulator_data(simulator):
    """
    Extract the three arrays needed for fit_persistence from a loaded
    SchwingerSimulation instance.

    Returns
    -------
    evolution_data : pd.DataFrame
        Main observables DataFrame (index = Time, column "Persistence" required).
    evolution_error : pd.DataFrame
        Per-point statistical error DataFrame (same shape as evolution_data).
    initial_state : QuantumCircuit
        Vacuum state circuit.
    trotter_error : np.ndarray | None
        Per-point Trotter error on "Persistence", estimated as |dt - dt/2|.
        None if only one Trotter configuration is available.
    """
    evolution_data  = simulator.evolution_data
    evolution_error = simulator.evolution_error
    initial_state   = simulator.initial_state

    # ── Trotter error estimation ────────────────────────────────────────────
    # Requires two Trotter configs in all_trotter_evolution_data.
    # Coarser step = larger step_multiplier; finer = smaller.
    trotter_error = None
    all_data = getattr(simulator, "all_trotter_evolution_data", {})

    if len(all_data) >= 2:
        # Sort configs by step_multiplier to get coarse/fine pair
        trotter_configs = simulator.config.get(
            "Temporal Evolution", {}
        ).get("Trotter_Steps", {})
        keys_sorted = sorted(
            [k for k in all_data if k != "mitigated"],
            key=lambda k: trotter_configs.get(k, {}).get("step_multiplier", 1.0),
            reverse=True,
        )
        coarse_key, fine_key = keys_sorted[0], keys_sorted[-1]

        if "Persistence" in all_data[coarse_key].columns and \
           "Persistence" in all_data[fine_key].columns:
            p_coarse = all_data[coarse_key]["Persistence"].values
            p_fine   = all_data[fine_key]["Persistence"].values
            # Interpolate coarse onto fine time grid if necessary
            if len(p_coarse) != len(p_fine):
                t_coarse = all_data[coarse_key].index.values
                t_fine   = all_data[fine_key].index.values
                p_coarse = np.interp(t_fine, t_coarse, p_coarse)
            trotter_error = np.abs(p_coarse - p_fine)

    return evolution_data, evolution_error, initial_state, trotter_error


def _build_total_sigma(evolution_data, evolution_error, trotter_error):
    """
    Combine statistical (shot) error and Trotter error in quadrature
    to produce the total per-point uncertainty on Persistence.

    sigma_total = sqrt(sigma_shots^2 + sigma_trotter^2)

    Returns np.ndarray of the same length as evolution_data, or None
    if no error information is available.
    """
    sigma_shots   = None
    sigma_trotter = None

    if evolution_error is not None and "Persistence" in evolution_error.columns:
        vals = evolution_error["Persistence"].values
        if np.any(vals > 0):
            sigma_shots = vals.astype(float)

    if trotter_error is not None:
        sigma_trotter = trotter_error.astype(float)
        # Align lengths (trotter may have been computed on a finer grid)
        if sigma_shots is not None and len(sigma_trotter) != len(sigma_shots):
            t_orig   = evolution_data.index.values
            t_trot   = np.linspace(t_orig[0], t_orig[-1], len(sigma_trotter))
            sigma_trotter = np.interp(t_orig, t_trot, sigma_trotter)

    if sigma_shots is not None and sigma_trotter is not None:
        return np.sqrt(sigma_shots**2 + sigma_trotter**2)
    elif sigma_shots is not None:
        return sigma_shots
    elif sigma_trotter is not None:
        return sigma_trotter
    else:
        return None

def _load_ideal_evolution_data(qubit_num_values, config, use_simulated_data=True):
    analysis_name = "decay_rate_qubits_num"
    file_name = "decay_rate_qubits_num"

    # Get needed data for each L
    analysis_name = "qubits_num_quench"
    ideal_qubits_num_data = load_evolution_and_initial(
        analysis_name,
        qubit_num_values,
        evolution_temp="qubits_num_{value}_quench_data.csv",
        initial_state_temp="qubits_num_{value}_initial_state.qpy",
        backup_config=config,
        backup_key="L",
        use_simulated_data=use_simulated_data,
    )

    return ideal_qubits_num_data

# ──────────────────────────────────────────────────────────────────────────── #
#  fit_persistence wrappers                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

def fit_persistence_qubits_num(config, qubits_num_values, simulators_data):
    """
    Fit persistence for a sweep of lattice sizes L.

    Parameters
    ----------
    config : dict
        Base simulation config (used as fallback if simulator config is missing).
    qubits_num_values : array-like
        Values of L (number of qubits) to analyse.
    simulators_data : dict
        Output of load_simulator_instance: {L: {"simulator": SchwingerSimulation}}.

    Returns
    -------
    pd.DataFrame
        Index = Qubits_Num, columns: Gamma_Simulated, Gamma_Err,
        Gamma_Analytical, E_Physical, Cut_Off_Times.
    """
    fit_results = {
        qubits_num: {
            "Gamma_Simulated": None,
            "Gamma_Err":       None,
            "Gamma_Analytical": None,
            "E_Physical":      None,
            "Cut_Off_Times":   None,
        }
        for qubits_num in qubits_num_values
    }

    for qubits_num in qubits_num_values:
        simulator = simulators_data.get(qubits_num, {}).get("simulator")
        if simulator is None:
            print(f"{getTimer()} WARNING: No simulator for L={qubits_num}, skipping.")
            continue

        evolution_data, evolution_error, initial_state, trotter_error = \
            _get_simulator_data(simulator)
        sigma = _build_total_sigma(evolution_data, evolution_error, trotter_error)

        # Use config stored in simulator, falling back to the provided one
        sim_config = copy.deepcopy(getattr(simulator, "config", config))
        sim_config["QubitsNumber"] = qubits_num
        sim_config["Hamiltonian"]["Parameters"]["L"] = qubits_num

        simulated_gamma, gamma_analytical, gamma_err, gamma_analytical_err, eE_evol, cut_off_times = \
            fit_persistence(
                evolution_data,
                sim_config,
                initial_state,
                sigma=sigma,
                evolution_error=evolution_error,
                use_offset=False,
                print_info=False,
            )

        fit_results[qubits_num]["Gamma_Simulated"]  = simulated_gamma
        fit_results[qubits_num]["Gamma_Err"]        = gamma_err
        fit_results[qubits_num]["Gamma_Analytical"] = gamma_analytical
        fit_results[qubits_num]["Gamma_Analytical_Err"] = gamma_analytical_err
        fit_results[qubits_num]["E_Physical"]       = eE_evol
        fit_results[qubits_num]["Cut_Off_Times"]    = cut_off_times

    fit_results_df = pd.DataFrame.from_dict(fit_results, orient="index")
    fit_results_df.index.name = "Qubits_Num"
    return fit_results_df


def fit_persistence_e0(config, e0_values, simulators_data):
    """
    Fit persistence for a sweep of background field values ε₀.

    Parameters
    ----------
    config : dict
        Base simulation config.
    e0_values : array-like
        Values of ε₀ to analyse.
    simulators_data : dict
        Output of load_simulator_instance: {e0: {"simulator": SchwingerSimulation}}.

    Returns
    -------
    pd.DataFrame
        Index = e0, columns: Gamma_Simulated, Gamma_Err,
        Gamma_Analytical, E_Physical.
    """
    fit_results = {
        e0: {
            "Gamma_Simulated":  None,
            "Gamma_Err":        None,
            "Gamma_Analytical": None,
            "E_Physical":       None,
        }
        for e0 in e0_values
    }

    for e0 in e0_values:
        simulator = simulators_data.get(e0, {}).get("simulator")
        if simulator is None:
            print(f"{getTimer()} WARNING: No simulator for e0={e0:.2f}, skipping.")
            continue

        evolution_data, evolution_error, initial_state, trotter_error = \
            _get_simulator_data(simulator)
        sigma = _build_total_sigma(evolution_data, evolution_error, trotter_error)

        sim_config = copy.deepcopy(getattr(simulator, "config", config))
        sim_config["Temporal Evolution"]["Quench"]["Parameters_to_Change"]["e0"] = e0

        simulated_gamma, gamma_analytical, gamma_err, gamma_analytical_err, eE_evol, cut_off_times = \
            fit_persistence(
                evolution_data,
                sim_config,
                initial_state,
                sigma=sigma,
                evolution_error=evolution_error,
                use_offset=False,
                print_info=False,
            )

        E_field_columns = [col for col in evolution_error.columns if col.startswith("E_link")]
        E_array = np.array([evolution_data.iloc[0][col] for col in E_field_columns]) # Assuming error is constant
        delta_E_array = np.array([evolution_error.iloc[0][col] for col in E_field_columns]) # Assuming error is constant

        # Store results
        fit_results[e0]["Gamma_Simulated"]  = simulated_gamma
        fit_results[e0]["Gamma_Err"]        = gamma_err
        fit_results[e0]["Gamma_Analytical"] = gamma_analytical
        fit_results[e0]["Gamma_Analytical_Err"] = gamma_analytical_err
        fit_results[e0]["E_Physical"]       = E_array
        fit_results[e0]["E_Physical_Err"]   = delta_E_array

    fit_results_df = pd.DataFrame.from_dict(fit_results, orient="index")
    fit_results_df.index.name = "e0"
    return fit_results_df


# ──────────────────────────────────────────────────────────────────────────── #
#  Main analysis script                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":

    # ── Config ────────────────────────────────────────────────────────────
    config = loadJsonConfig("SchwingerSimulation_Persistence_v2.json")
    USE_SIMULATED_DATA = True

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Γ vs number of qubits — finite-size scaling
    # ═══════════════════════════════════════════════════════════════════════
    print(f"{getTimer()} INFO: Starting persistence analysis...")
    print("\n1. Γ vs number of qubits — finite-size scaling\n")
    if True:
        analysis_name    = "01_Persistence_Noisy_Data"
        file_name        = "01_01_DecayRate_vs_QubitsNum"
        min_qubit_num, max_qubit_num, step = 10, 20, 2
        min_qubit_num, max_qubit_num, step = 10, 18, 2
        qubit_num_values = np.arange(min_qubit_num, max_qubit_num + step, step)
        qubit_num_values = np.array([int(qubit_num) for qubit_num in qubit_num_values])
        
        # Load ideal evolution data for comparision with noisy
        ideal_qubits_num_data = _load_ideal_evolution_data(qubit_num_values, config,
                                                           use_simulated_data=USE_SIMULATED_DATA)
        vqe_initial_states = {
            qubits_num: ideal_qubits_num_data[qubits_num]["initial_state"]
            for qubits_num in qubit_num_values
        }

        # Load .pkl simulator instances (one per L)
        qubits_num_simulators = load_simulator_instance(
            analysis_name,
            qubit_num_values,
            simulator_file_template="Sim_Instance_{value}_Qubits.pkl",
            backup_config=config,
            backup_key="L",
            use_simulated_data=USE_SIMULATED_DATA,
            initial_states=vqe_initial_states
        )

        fit_results_df = fit_persistence_qubits_num(
            config, qubit_num_values, qubits_num_simulators
        )

        plot_params = parseDictToPlot(
            {
                **config["Hamiltonian"]["Parameters"],
                **config["Temporal Evolution"]["Quench"]["Parameters_to_Change"],
            },
            remove_keys=["L"],
            rename_keys={"e0": "$\\varepsilon_0$"},
        )
        fig, ax = plot_gamma_vs_qubitNum(fit_results_df, params=plot_params)
        save_data(fig, analysis_name, f"{file_name}.png", rootPath=plt_folder, dpi=300)
        save_data(fit_results_df, analysis_name, f"{file_name}.csv", rootPath=plt_folder)

        # Find best L (minimum deviation to analytical)
        fit_results_df["Deviation"] = np.abs(
            1 - fit_results_df["Gamma_Analytical"] / fit_results_df["Gamma_Simulated"]
        )
        best_qubit_num = int(fit_results_df["Deviation"].idxmin())
        print(f"{getTimer()} INFO: Best qubit number: {best_qubit_num}")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Best fit — exponential decay plot with error band
    # ═══════════════════════════════════════════════════════════════════════
    print("\n2. Best fit — exponential decay plot with error band\n")
    if True:
        analysis_name = f"01_02_BestDecayRate_vs_QubitsNum"
        file_name     = f"01_02_BestDecayRate_vs_QubitsNum"

        best_simulator = qubits_num_simulators[best_qubit_num]["simulator"]
        evolution_data, evolution_error, initial_state, trotter_error = \
            _get_simulator_data(best_simulator)
        sigma = _build_total_sigma(evolution_data, evolution_error, trotter_error)

        best_config = copy.deepcopy(
            getattr(best_simulator, "config", config)
        )
        best_config["QubitsNumber"] = best_qubit_num
        best_config["Hamiltonian"]["Parameters"]["L"] = best_qubit_num

        fig, axes = fit_persistence(
            evolution_data,
            best_config,
            initial_state,
            sigma=sigma,
            evolution_error=evolution_error,
            use_offset=False,
            return_plot=True,
        )
        save_data(fig, analysis_name, f"{file_name}.png", rootPath=plt_folder, dpi=300)
        save_data(fit_results_df, analysis_name, f"{file_name}.csv", rootPath=plt_folder)

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Persistence vs time — full evolution with regimes
    # ═══════════════════════════════════════════════════════════════════════
    print("\n3. Persistence vs time — full evolution with regimes\n")
    if True:
        analysis_name = f"01_03_Persistence_vs_Time"
        file_name     = f"01_03_Persistence_vs_Time"

        cut_off_times = fit_results_df.loc[best_qubit_num, "Cut_Off_Times"]
        plot_params   = parseDictToPlot(
            {
                **best_config["Hamiltonian"]["Parameters"],
                **best_config["Temporal Evolution"]["Quench"]["Parameters_to_Change"],
            },
            remove_keys=[],
            rename_keys={"e0": "$\\varepsilon_0$"},
        )

        # Pass error band (total sigma) to the regime plot if supported
        fig, ax = plot_persistenece_vs_time_regimes(
            evolution_data,
            cut_off_times,
            plot_params,
            sigma=sigma,          # band on the persistence curve
            ideal_evolution_data=ideal_qubits_num_data.get(best_qubit_num, {}).get("evolution_data"),
        )
        save_data(fig, analysis_name, f"{file_name}.png", rootPath=plt_folder, dpi=300)
        save_data(fit_results_df, analysis_name, f"{file_name}.csv", rootPath=plt_folder)

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Γ vs ε₀ sweep
    # ═══════════════════════════════════════════════════════════════════════
    print("\n4. Γ vs ε₀ sweep\n")
    if True:
        qubits_num    = best_qubit_num
        analysis_name = f"01_04_Gamma_vs_E0"
        file_name     = f"01_04_Gamma_vs_E0"

        # Weak field regime
        e0_values = np.arange(0.20, 0.36 + 0.04, 0.04)
        # Schwinger + strong field regime
        e0_values = np.concatenate((e0_values, np.arange(0.40, 0.98 + 0.02, 0.02)))

        # Load one simulator per e0 value
        # The initial state is the same for all (e0 is quenched, not a prep param)
        e0_simulators = load_simulator_instance(
            analysis_name,
            e0_values,
            simulator_file_template="Sim_Instance_e0_{value}"+f"_{best_qubit_num}_Qubits.pkl",
            backup_config=best_config,
            backup_key="e0",
            backup_key_is_quench=True,
            use_simulated_data=USE_SIMULATED_DATA,
            initial_states={e0: vqe_initial_states[best_qubit_num] for e0 in e0_values}
        )

        fit_results_df = fit_persistence_e0(best_config, e0_values, e0_simulators)

        plot_params = parseDictToPlot(
            best_config["Hamiltonian"]["Parameters"],
            remove_keys=["e0"],
            rename_keys={},
        )
        fig, ax = plot_gamma_vs_e0(fit_results_df, params=plot_params)
        save_data(fig, analysis_name, f"{file_name}.png", rootPath=plt_folder, dpi=300)
        save_data(fit_results_df, analysis_name, f"{file_name}.csv", rootPath=plt_folder)

    # ═══════════════════════════════════════════════════════════════════════
    # 5. log(Γ/⟨E⟩) vs 1/⟨E⟩ — non-perturbative Schwinger signature
    # ═══════════════════════════════════════════════════════════════════════
    print("\n5. log(Γ/⟨E⟩) vs 1/⟨E⟩ — non-perturbative Schwinger signature\n")
    if True:
        analysis_name = f"01_05_LogPersistence_vs_ElectricField"
        file_name     = f"01_05_LogPersistence_vs_ElectricField"

        # Filter to Schwinger regime (deviation ≤ 20 %)
        max_deviation = 0.20
        fit_results_df["Dev"] = (
            1 - fit_results_df["Gamma_Analytical"] / fit_results_df["Gamma_Simulated"]
        )
        e0_schwinger = fit_results_df[
            np.abs(fit_results_df["Dev"]) <= max_deviation
        ].index.values
        e0_reg = e0_schwinger  # already sorted

        fit_reg_df     = fit_results_df.loc[e0_reg]
        gamma_sim      = fit_reg_df["Gamma_Simulated"].values
        gamma_err_vals = fit_reg_df["Gamma_Err"].values          # propagate to plot
        field_values   = fit_reg_df["E_Physical"].apply(np.mean).values
        field_err_values = fit_reg_df["E_Physical_Err"].apply(lambda x: np.sum(x**2)**0.5).values / len(fit_reg_df["E_Physical"].iloc[0])

        plot_params = parseDictToPlot(
            best_config["Hamiltonian"]["Parameters"],
            remove_keys=["e0"],
            rename_keys={},
        )
        fig, axes, fit_params = plot_gamma_vs_electricField(
            gamma_sim, e0_reg, field_values,
            gamma_err=gamma_err_vals,      # pass if your plot function supports it
            field_err=field_err_values,
            params=plot_params
        )
        save_data(fig, analysis_name, f"{file_name}.png", rootPath=plt_folder, dpi=300)
        save_data(fit_reg_df, analysis_name, f"{file_name}.csv", rootPath=plt_folder)

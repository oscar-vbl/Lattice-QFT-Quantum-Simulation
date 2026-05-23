"""
Basic validation results of the architecture
"""

import os

# Use one thread per process
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import concurrent.futures

import sys
import contextlib
import numpy as np
import copy
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
import scipy.sparse.linalg as sla
from qiskit import transpile
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.as_posix())
from _config import PLOTS_FOLDER as plt_folder
from _config import DATA_FOLDER as data_folder
from Utils import getTimer, save_data, loadJsonConfig
from SchwingerSimulation import SchwingerSimulation
from ResultsAnalysis import calculate_global_gradient_variance
from Plots import (
    plot_validation_ansatzes,
    plot_duration_ansatzes,
    plot_computational_resources,
    plot_variance_decay,
    plot_ansatz_circuit,
)

from R00_ResultsCommon import load_simulator_instance, get_simulation_data


# Avoid individual run prints
def quiet_get_simulation_data(qubits_num_config):
    """
    Run get_simulation_data redirecting all prints to nothing.
    """
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            return get_simulation_data(qubits_num_config)


def get_ansatz_comparison(
    config,
    qubits_num_values,
    runs=5,
    parallel_runs=None,
    top_runs=5,
    analysis_name="",
    use_simulated_data=True,
    save_simulated_data=True,
):

    comparisons = {
        qubits_num: {
            "Overlap_Values": [],
            "Overlap_Mean": None,
            "Overlap_Error": None,
            "Sim_Energy_Values": [],
            "Sim_Energy_Mean": None,
            "Sim_Energy_Error": None,
            "Num_Energy_Value": None,
            "Dev_Energy_Values": [],
            "Dev_Energy_Mean": None,
            "Dev_Energy_Error": None,
            "Time_Values": [],
            "Time_Mean": None,
            "Time_Error": None,
            "Optimization_History": [],
        }
        for qubits_num in qubits_num_values
    }

    ansatz_type = config["Ansatz"]["Type"]

    for qubits_num in qubits_num_values:
        qubits_num_config = copy.deepcopy(config)
        qubits_num_config["QubitsNumber"] = qubits_num
        qubits_num_config["Hamiltonian"]["Parameters"]["L"] = qubits_num

        total_runs = runs  # Big number of attempts
        top_k = top_runs  # Take only best runs
        run_results = []

        simulator_save_temp = (
            f"Simulator_{ansatz_type}_{qubits_num}Qubits" + "_Run{value}.pkl"
        )
        if use_simulated_data:
            # Load simulator data if available
            runs_simulations = load_simulator_instance(
                analysis_name, list(range(total_runs)), simulator_save_temp
            )
            sims_are_run = all(
                [
                    runs_simulations[run]["simulator"] is not None
                    for run in runs_simulations
                ]
            )
        else:
            sims_are_run = False

        if not sims_are_run:
            print(
                f"{getTimer()} INFO: Ansatz {ansatz_type} for L={qubits_num}. Starting {total_runs} parallel runs..."
            )

            # Create Processes Pool with max number of parallel runs
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=parallel_runs
            ) as executor:
                # 1. Send all runs to Process Pool
                futures = [
                    executor.submit(quiet_get_simulation_data, qubits_num_config)
                    for _ in range(total_runs)
                ]

                # 2. Get data as soon as they finish
                for future in concurrent.futures.as_completed(futures):
                    try:
                        simulator, duration = future.result()

                        # Get simulator instance and vacuum energy (to sort the results)
                        energy = simulator.vacuum_energy

                        run_results.append(
                            {
                                "simulator": simulator,
                                "energy": energy,
                                "duration": duration,
                            }
                        )
                        print(
                            f"{getTimer()} INFO: Run completed for {qubits_num} qubits in {duration:.2f} s, with energy: {energy:.4f}"
                        )

                    except Exception as exc:
                        print(
                            f"{getTimer()} WARNING: Exception raised by run for {qubits_num} qubits: {exc}"
                        )

            # Get original prints again
            sys.stdout = sys.__stdout__

            # Order all results based on energy, from lowest to highest
            # Due to variational theorem, the smallest the better
            sorted_results = sorted(run_results, key=lambda x: x["energy"])

            if save_simulated_data:
                for run in range(len(sorted_results)):
                    save_data(
                        sorted_results[run]["simulator"],
                        analysis_name,
                        simulator_save_temp.format(value=run),
                        rootPath=data_folder,
                    )

            # Get only first top_k
            # Some may fall in valleys of excited states
            best_runs = sorted_results[:top_k]
        else:
            best_runs = [
                {
                    "simulator": runs_simulations[run]["simulator"],
                    "energy": runs_simulations[run]["simulator"].vacuum_energy,
                    "duration": runs_simulations[run]["simulator"].vqe_duration,
                }
                for run in runs_simulations
            ]

        for run in range(top_k):
            # Get simulator instance
            simulator, duration = (
                best_runs[run]["simulator"],
                best_runs[run]["duration"],
            )
            print(
                f"{getTimer()} INFO: Simulation for {qubits_num} qubits took {(duration):.2f} seconds."
            )
            initial_state = Statevector(simulator.initial_state).data
            vacuum_energy = simulator.vacuum_energy
            vqe_duration = simulator.vqe_duration

            if run == 0:
                ham_sparse = simulator.hamiltonian_prep.to_matrix(sparse=True)
                eigvals, eigvecs = sla.eigsh(ham_sparse, k=1, which="SA")
                num_gs = eigvecs[:, 0]
                num_energy = eigvals[0]

                comparisons[qubits_num]["Num_Energy_Value"] = num_energy

            comparisons[qubits_num]["Overlap_Values"].append(
                np.abs(np.vdot(num_gs, initial_state)) ** 2
            )
            comparisons[qubits_num]["Sim_Energy_Values"].append(vacuum_energy)
            comparisons[qubits_num]["Dev_Energy_Values"].append(
                (vacuum_energy - num_energy) / np.abs(num_energy)
            )
            comparisons[qubits_num]["Time_Values"].append(vqe_duration)
            comparisons[qubits_num]["Optimization_History"].append(
                simulator.optimization_history
            )

        print(
            f"{getTimer()} INFO: Overlap for {qubits_num} qubits:",
            np.mean(comparisons[qubits_num]["Overlap_Values"]),
        )
        print(
            f"{getTimer()} INFO: Energy error for {qubits_num} qubits:",
            np.mean(comparisons[qubits_num]["Dev_Energy_Values"]),
        )

        comparisons[qubits_num]["Overlap_Mean"] = np.mean(
            comparisons[qubits_num]["Overlap_Values"]
        )
        comparisons[qubits_num]["Overlap_Error"] = np.std(
            comparisons[qubits_num]["Overlap_Values"]
        )

        comparisons[qubits_num]["Sim_Energy_Mean"] = np.mean(
            comparisons[qubits_num]["Sim_Energy_Values"]
        )
        comparisons[qubits_num]["Sim_Energy_Error"] = np.std(
            comparisons[qubits_num]["Sim_Energy_Values"]
        )

        comparisons[qubits_num]["Dev_Energy_Mean"] = np.mean(
            comparisons[qubits_num]["Dev_Energy_Values"]
        )
        comparisons[qubits_num]["Dev_Energy_Error"] = np.std(
            comparisons[qubits_num]["Dev_Energy_Values"]
        )

        comparisons[qubits_num]["Time_Mean"] = np.mean(
            comparisons[qubits_num]["Time_Values"]
        )
        comparisons[qubits_num]["Time_Error"] = np.std(
            comparisons[qubits_num]["Time_Values"]
        )

    return comparisons


if __name__ == "__main__":
    # Load Config
    config = loadJsonConfig("SchwingerSimulation_Validation.json")

    analysis_name = "00_Ansatz Precision and Duration"

    ansatzes = ["HVA", "HVA Simple", "ExcitationPreserving"]

    # 1. Plot circuits
    if True:
        L = 6
        reps = 1
        for ansatz in ansatzes:
            fig = plot_ansatz_circuit(config, L, reps, ansatz)
            # Save circuit
            file_name = f"00_01_Circuit_{ansatz}"
            save_data(
                fig, analysis_name, f"{file_name}.png", rootPath=plt_folder, dpi=300
            )
            save_data(
                fig, analysis_name, f"{file_name}.pdf", rootPath=plt_folder, dpi=300
            )

    # 2. Overlap vs Ansatz reps
    if True:
        reps_num = [1, 2, 3, 4]
        qubits_num = 8  # Medium value
        ansatz_config = copy.deepcopy(config)
        ansatz_config["QubitsNumber"] = qubits_num
        ansatz_config["Hamiltonian"]["Parameters"]["L"] = qubits_num
        ansatzes_comparison = {}
        for ansatz in ansatzes:
            ansatz_config["Ansatz"]["Type"] = ansatz
            ansatz_comparison = {}
            for reps in reps_num:
                analysis_name_reps = f"{analysis_name}_{reps}_Reps"
                reps_config = copy.deepcopy(ansatz_config)
                reps_config["Ansatz"]["Reps"] = reps
                comparisons = get_ansatz_comparison(
                    reps_config, [qubits_num], analysis_name=analysis_name_reps
                )
                if reps != qubits_num:
                    comparisons[reps] = comparisons[qubits_num]
                    del comparisons[qubits_num]
                ansatz_comparison = {**ansatz_comparison, **comparisons}
            ansatzes_comparison[ansatz] = ansatz_comparison
        # Infidelity ($1 - \text{Overlap}$) in logarithmic scale on Y axis.
        fig_err, axes = plot_validation_ansatzes(
            reps_num, ansatzes_comparison, x_label="Ansatz Repetitions"
        )
        fig_dur, ax = plot_duration_ansatzes(
            reps_num, ansatzes_comparison, x_label="Ansatz Repetitions"
        )
        # Save plot
        file_name = "00_02_Overlap_Vs_Reps"
        save_data(
            fig_err,
            analysis_name,
            f"{file_name}_Errors.png",
            rootPath=plt_folder,
            dpi=300,
        )
        save_data(
            fig_err,
            analysis_name,
            f"{file_name}_Errors.pdf",
            rootPath=plt_folder,
            dpi=300,
        )
        save_data(
            fig_dur,
            analysis_name,
            f"{file_name}_Duration.png",
            rootPath=plt_folder,
            dpi=300,
        )
        save_data(
            fig_dur,
            analysis_name,
            f"{file_name}_Duration.pdf",
            rootPath=plt_folder,
            dpi=300,
        )
        save_data(
            ansatz_comparison, analysis_name, f"{file_name}.pkl", rootPath=plt_folder
        )

    # 3. Overlap vs Qubits number ($L$)
    if True:
        qubits_nums = [4, 6, 8, 10, 12, 14, 16]
        ansatzes = ["HVA", "HVA Simple", "ExcitationPreserving"]
        ansatz_comparison = {}
        for ansatz in ansatzes:
            ansatz_config = copy.deepcopy(config)
            ansatz_config["Ansatz"]["Type"] = ansatz
            comparisons = get_ansatz_comparison(
                ansatz_config,
                qubits_nums,
                runs=15,
                parallel_runs=5,
                analysis_name=analysis_name,
            )
            ansatz_comparison = {**ansatz_comparison, **{ansatz: comparisons}}
        # Infidelity ($1 - \text{Overlap}$) in logarithmic scale on Y axis.
        fig_err, axes = plot_validation_ansatzes(qubits_nums, ansatz_comparison)
        fig_dur, ax = plot_duration_ansatzes(qubits_nums, ansatz_comparison)

        # Save plot
        file_name = "00_03_Overlap_Vs_Qubits"
        save_data(
            fig_err,
            analysis_name,
            f"{file_name}_Errors.png",
            rootPath=plt_folder,
            dpi=300,
        )
        save_data(
            fig_err,
            analysis_name,
            f"{file_name}_Errors.pdf",
            rootPath=plt_folder,
            dpi=300,
        )
        save_data(
            fig_dur,
            analysis_name,
            f"{file_name}_Duration.png",
            rootPath=plt_folder,
            dpi=300,
        )
        save_data(
            fig_dur,
            analysis_name,
            f"{file_name}_Duration.pdf",
            rootPath=plt_folder,
            dpi=300,
        )
        save_data(
            ansatz_comparison, analysis_name, f"{file_name}.pkl", rootPath=plt_folder
        )

    # 4. Computational resources
    if True:
        # Native IBM Quantum Gates (Eagle/Heron)
        ibm_basis_gates = ["cx", "rz", "sx", "x"]

        L_vals = [4, 6, 8, 10]
        ansatzes = ["HVA", "HVA Simple", "ExcitationPreserving"]

        depths, cnots = {}, {}

        for ansatz in ansatzes:
            ansatz_config = copy.deepcopy(config)
            ansatz_config["Ansatz"]["Type"] = ansatz

            depths[ansatz] = {}
            cnots[ansatz] = {}
            for L in L_vals:
                ansatz_config["QubitsNumber"] = L
                ansatz_config["Hamiltonian"]["Parameters"]["L"] = L
                sim = SchwingerSimulation(copy.deepcopy(ansatz_config))
                sim.hamiltonian_prep = sim.get_hamiltonian()
                ansatz_circuit = sim.get_ansatz()

                # Transpile to level 2 (medium optimization, Qiskit standard)
                transpiled_circ = transpile(
                    ansatz_circuit, basis_gates=ibm_basis_gates, optimization_level=2
                )

                # Extract metrics
                depths[ansatz][L] = transpiled_circ.depth()
                cnots[ansatz][L] = transpiled_circ.count_ops().get("cx", 0)

        # Plot resources (CNOTs and depth)
        fig_cost, axes_cost = plot_computational_resources(
            ansatzes, L_vals, depths, cnots
        )
        # Save plot
        file_name = "00_04_ComputationalResources"
        save_data(
            fig_cost, analysis_name, f"{file_name}.png", rootPath=plt_folder, dpi=300
        )
        save_data(
            fig_cost, analysis_name, f"{file_name}.pdf", rootPath=plt_folder, dpi=300
        )
        save_data(
            {"depths": depths, "cnots": cnots},
            analysis_name,
            f"{file_name}.pkl",
            rootPath=plt_folder,
        )

    # 5. Barren plateaus
    if True:
        L_vals = [4, 6, 8, 10, 12]
        var_grads = {}

        for ansatz in ansatzes:
            ansatz_config = copy.deepcopy(config)
            ansatz_config["Ansatz"]["Type"] = ansatz

            var_grads[ansatz] = {}
            for L in L_vals:
                ansatz_config["QubitsNumber"] = L
                ansatz_config["Hamiltonian"]["Parameters"]["L"] = L
                sim = SchwingerSimulation(copy.deepcopy(ansatz_config))
                sim.hamiltonian_prep = sim.get_hamiltonian()
                ansatz_circuit = sim.get_ansatz()

                # Sparse hamiltonian
                H_sparse = sim.hamiltonian_prep.to_matrix(sparse=True)

                # Calculate variances with 100 random samples
                ansatz_var = calculate_global_gradient_variance(
                    ansatz_circuit, H_sparse, num_samples=100
                )

                # Extract metrics
                var_grads[ansatz][L] = ansatz_var

        # Plot variance decay
        fig_bp, ax = plot_variance_decay(ansatzes, L_vals, var_grads)
        # Save plot
        file_name = "00_05_BarrenPlateaus"
        save_data(
            fig_cost, analysis_name, f"{file_name}.png", rootPath=plt_folder, dpi=300
        )
        save_data(
            fig_cost, analysis_name, f"{file_name}.pdf", rootPath=plt_folder, dpi=300
        )
        save_data(var_grads, analysis_name, f"{file_name}.pkl", rootPath=plt_folder)

    plt.show()

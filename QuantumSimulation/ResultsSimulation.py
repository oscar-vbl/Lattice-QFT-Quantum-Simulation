from SchwingerSimulation import SchwingerSimulation
from Plots import simplePlot
from Utils import loadJsonConfig, getValidFileName
from Utils import PLOTS_FOLDER as plt_folder
import time
import numpy as np
import matplotlib.pyplot as plt
import os

def persistanceQubitNumDependant(config, qubits_nums):
    evolution_data = {}
    for qubits_num in qubits_nums:
        config["QubitsNumber"] = qubits_num
        config["Hamiltonian"]["Parameters"]["L"] = qubits_num
        start = time.time()
        simulator = SchwingerSimulation(config)
        simulator.start_simulation()
        end = time.time()
        print(f"Simulation for {qubits_num} qubits took {end - start:.2f} seconds.")
        evolution_data[qubits_num] = simulator.evolution_data

    # Plot all persistences (assumes `evolution_data` is a dict qubits_num -> DataFrame)
    plt.figure(figsize=(8, 5))
    keys = sorted(k for k in evolution_data.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(keys)))
    for i, qnum in enumerate(keys):
        df = evolution_data[qnum]
        if df is None or "Persistance" not in df.columns:
            continue
        t = df.index
        y = df["Persistance"]
        plt.plot(t, y, label=f"{qnum} qubits", color=colors[i], linewidth=1.2, marker=None)

    plt.xlabel("Time")
    plt.ylabel("Vacuum Persistence")
    plt.title("Vacuum Persistence vs Time (different system sizes)")
    plt.grid(True)
    plt.legend(title="System size")
    plt.tight_layout()
    plt.show()

def plotVaccumPersistance(config):
    simulator = SchwingerSimulation(config)
    simulator.start_simulation()
    save_path = os.path.join(plt_folder, f"vaccum_persistance_{simulator.qubits_num}qubits.png")
    save_path = getValidFileName(save_path)
    x = list(simulator.evolution_data.index)
    y = simulator.evolution_data["Persistance"]
    x_label = "Time"
    y_label = "Vaccum Persistance"
    total_time = simulator.config["Temporal Evolution"]["Total_Time"]
    title   = f"Vaccum Persistance for {simulator.qubits_num} qubits and total time {total_time}"
    fig, ax = simplePlot(x, y, title=title, xlabel=x_label, ylabel=y_label, savePath=save_path)
    #import matplotlib.pyplot as plt
    #plt.show()


if __name__ == "__main__":
    config = loadJsonConfig("SchwingerSimulation_v0.json")
    qubits_nums = [6,8,10,12]
    qubits_nums = [6,8,10]
    persistanceQubitNumDependant(config, qubits_nums)

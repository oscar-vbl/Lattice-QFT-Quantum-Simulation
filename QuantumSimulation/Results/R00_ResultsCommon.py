'''
Reusable functions for results analysis and plotting.
'''
import sys
import numpy as np
import time
import copy
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())
from _config import PLOTS_FOLDER as plt_folder
from _config import DATA_FOLDER  as data_folder
from Utils import getTimer, load_data, save_data
from SchwingerSimulation import SchwingerSimulation

def get_simulation_data(config, initial_state=None):
    start = time.time()
    simulator = SchwingerSimulation(config, initial_state=initial_state)
    simulator.run_simulation()
    end = time.time()

    return simulator, end-start

def load_evolution_and_initial(analysis_name, values,
                           evolution_temp, initial_state_temp,
                           use_simulated_data=True,
                           backup_config=None,
                           backup_initial_state=None,
                           backup_key="L",
                           backup_key_is_quench=False,
                           save_if_simulated=True):
    values_data = {}
    for value in values:
        if isinstance(value, int) or isinstance(value, np.integer):
            fileValue = str(value)
        else:
            fileValue = f"{value:.2f}"

        if use_simulated_data:
            # Load previously simulated data
            try:
                # Load evolution data
                evolution_data = load_data(analysis_name, evolution_temp.format(value=fileValue), indexSet="Time")
                # Load initial state
                initial_state  = load_data(analysis_name,  initial_state_temp.format(value=fileValue))
                # Store in dict
                values_data[value] = {
                    "evolution_data": evolution_data,
                    "initial_state": initial_state
                }
                data_is_loaded = True
            except FileNotFoundError:
                print(f"{getTimer()} WARNING: Data for value={value} not found.")
                data_is_loaded = False
        else:
            # Skip loading data to run simulation
            data_is_loaded = False
        
        if not data_is_loaded:
            # Run simulation if no previous data is loaded and config is provided
            if backup_config is not None:
                print(f"{getTimer()} INFO: Running simulation for value={value} with backup config.")
                value_config = copy.deepcopy(backup_config)
                if backup_key_is_quench:
                    value_config["Temporal Evolution"]["Quench"]["Parameters_to_Change"][backup_key] = value
                else:
                    if backup_key == "L" or backup_key == "QubitsNumber":
                        value_config["QubitsNumber"] = value
                        value_config["Hamiltonian"]["Parameters"]["L"] = value
                    else:
                        value_config["Hamiltonian"]["Parameters"][backup_key] = value
                simulator, duration = get_simulation_data(value_config, initial_state=backup_initial_state)
                evolution_data = simulator.evolution_data
                initial_state  = simulator.initial_state
                values_data[value] = {
                    "evolution_data": evolution_data,
                    "initial_state": initial_state
                }
                print(f"{getTimer()} INFO: Simulation for value={value} took {(duration):.2f} seconds.")
                if save_if_simulated:
                    # Save evolution data
                    save_data(evolution_data, analysis_name, evolution_temp.format(value=fileValue), rootPath=data_folder)
                    # Save initial state
                    save_data(initial_state,  analysis_name, initial_state_temp.format(value=fileValue), rootPath=data_folder)
            else:
                print(f"{getTimer()} ERROR: No backup config provided for value={value}. Skipping.")
        
    return values_data

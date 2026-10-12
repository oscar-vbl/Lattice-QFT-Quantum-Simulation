# QuantumSimulation/hardware/noise.py
from qiskit_aer.noise import NoiseModel, depolarizing_error
from QuantumSimulation.core.types import NoiseConfig
from QuantumSimulation.Utils import getTimer

def build_noise_model(config: NoiseConfig | None) -> NoiseModel | None:
    """
    Build a Qiskit Aer noise model from configuration.
    """
    if config is None or not config.errors:
        return None
        
    print(f"{getTimer()} INFO: Applying noise model to AerSimulator.")
    aer_noise_model = NoiseModel()
    
    for error_conf in config.errors:
        if error_conf.error_type.title() == "Depolarizing":
            error_param = depolarizing_error(error_conf.probability, error_conf.num_qubits)
            aer_noise_model.add_all_qubit_quantum_error(error_param, error_conf.gates)
        else:
            print(f"{getTimer()} WARNING: Noise model error type '{error_conf.error_type}' not recognized. Skipping.")
            
    return aer_noise_model
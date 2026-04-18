import sys
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())
from SchwingerSimulation import SchwingerSimulation
from Calculations import calculateEnergy
from Calculations import calculateVacuumPersistence

def test_full_simulation_pipeline_smoke_test():
    """Validates that a full simulation pipeline runs without throwing exceptions."""
    time_steps = 10
    test_config = {
        "QubitsNumber": 4, # Muy pequeño para que sea instantáneo
        "Hamiltonian": {
            "Type": "Schwinger", "Gauge": "Temporal",
            "Parameters": {"L": 4, "a": 0.5, "m": 0.1, "e0": 0.0}
        },
        "Backend": {"Type": "StatevectorEstimator"},
        "Ansatz": {
            "Type": "ExcitationPreserving", "Reps": 1,
            "Minimizer": {"Method": "COBYLA", "Options": {"maxiter": 2}} # Only 2 iters
        },
        "Temporal Evolution": {
            "Active": True,
            "Time_Steps": time_steps,
            "Total_Time": 10,
            "Quench": {
                "Active": True,
                "Parameters_to_Change": {
                    "e0": 0.5
                }
            },
            "Evolution_Gate": {
                "Type": "Pauli",
                "Synthesis": "TrotterSuzuki",
                "Synthesis_Params": {"order": 2}
            },
            "Evolution_Method": "MatrixExponential",
            "Observables": {
                "Observables_List": ["Energy", "Persistence", "Pair_Creation"],
                "Observables_Params": {
                    "Energy": {},
                    "Persistence": {}
                }
            }
        }
    }
    
    sim = SchwingerSimulation(test_config)
    sim.run_simulation()
    
    # Check results have been stored
    assert sim.evolution_data is not None, "DataFrame of results was not created."
    assert len(sim.evolution_data) == time_steps, f"Must have {time_steps} rows (t=0 + {time_steps-1} time steps)."
    assert "Persistence" in sim.evolution_data.columns


def test_estimator_v2_parity():
    """Assert EstimatorV2 and Statevector dan el mismo valor esperado."""
    # 1. Prepare simple circuit and observable
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1) # Estado de Bell
    
    observable = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 0.5)])
    
    # 2. Calculate with Statevector (no estimator)
    sv = Statevector(qc)
    energy_exact = calculateEnergy(sv, observable, estimator=None)
    
    # 3. Calculate with QuantumCircuit (EstimatorV2)
    estimator = StatevectorEstimator()
    energy_v2 = calculateEnergy(qc, observable, estimator=estimator)
    
    # 4. Validate results are close
    assert np.isclose(energy_exact, energy_v2, atol=1e-7), \
        f"Mismatch: Exact={energy_exact}, V2={energy_v2}"
    
def test_sampler_vacuum_persistence():
    sampler = StatevectorSampler()
    
    # Initial state: |00> with Hadamard on the first qubit to create superposition
    initial_circuit = QuantumCircuit(2)
    initial_circuit.h(0)
    
    # Case A: Null evolution (the state is the same)
    evolved_circuit_A = initial_circuit.copy()
    pers_A = calculateVacuumPersistence(evolved_circuit_A, initial_circuit, sampler)
    assert np.isclose(pers_A, 1.0, atol=1e-2), "The persistence should be 1 if there is no evolution."
    
    # Case B: Orthogonal evolution (we apply Z which changes the phase and breaks the overlap)
    evolved_circuit_B = initial_circuit.copy()
    evolved_circuit_B.z(0) 
    evolved_circuit_B.x(1) # Orthogonal state
    pers_B = calculateVacuumPersistence(evolved_circuit_B, initial_circuit, sampler)
    assert np.isclose(pers_B, 0.0, atol=1e-2), "The persistence should be 0 for orthogonal states."

if __name__ == "__main__":
    test_full_simulation_pipeline_smoke_test()
    print("test_full_simulation_pipeline_smoke_test passed.")
    
    test_estimator_v2_parity()
    print("test_estimator_v2_parity passed.")
    
    test_sampler_vacuum_persistence()
    print("test_sampler_vacuum_persistence passed.")

    print("\n✓ All tests passed!")

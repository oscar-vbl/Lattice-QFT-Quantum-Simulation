# QuantumSimulation

Technical documentation for quantum simulation modules.

## Core Modules

### `SchwingerSimulation.py`

The main orchestration class. Handles:
- Hamiltonian construction from config
- Ansatz circuit building
- VQE optimization for vacuum state
- Temporal evolution with observable tracking

**Key Methods**:
- `run_simulation()`: Execute full pipeline
- `get_hamiltonian()`: Build SparsePauliOp
- `get_ansatz()`: Construct parametrized circuit
- `get_vacuum()`: Optimize and return ground state

### `Operators.py`

Hamiltonian and observable definitions:
- `buildSchwingerHamiltonianTemporalGauge()`: Main Hamiltonian (temporal gauge, e=1)
- `gauss_operator()`: Charge constraint operator
- `measure_electric_field()`: Sample ⟨E⟩ from state

### `circuitBuilder.py`

Quantum circuit utilities:
- `buildCircuit()`: Create base circuit from config
- `addGate()`: Add parametrized gates
- Supports EfficientSU2, n_local, custom ansatzes

### `Calculations.py`

Observable calculations on quantum states:
- `calculateEnergy()`: ⟨ψ|H|ψ⟩
- `calculateVacuumPersistence()`: ⟨ψ|ψ₀⟩²
- `calculatePairCreation()`: Pair creation rate
- `checkChargeSymmetry()`: Validate U(1) constraint

### `Plots.py`

Visualization:
- `plot_simulated_vs_analytical()`: Persistence + Schwinger overlay
- `plot_persistenece_vs_time_regimes()`: Mark Zeno/Schwinger/revival regions
- `plot_gamma_vs_qubitNum()`: Decay rate scaling
- `plot_gamma_vs_e0()`: e0-field dependence

### `ResultsAnalysis.py`

Post-simulation analysis:
- `fit_persistence()`: Fit exponential decay identifying physical regimes
  - Automatically detects: Zeno (initial), Schwinger (exponential), Interference, Revivals
  - The fit focuses on the **Schwinger region** for Γ extraction.
- `check_regime()`: Verify simulation is in valid physical regime

### `Utils.py`

I/O and helpers:
- `loadJsonConfig()`: Parse configuration files
- `save_data()` / `load_data()`: Persist DataFrames and circuits
- `getTimer()`: Timestamped logging

### `_config.py`

Path configuration (uses `pathlib`):
- `PROJECT_ROOT`
- `PLOTS_FOLDER`, `DATA_FOLDER`, `CONFIGS_FOLDER`
- All paths relative to project root

## Configuration Files

### `Configs/`

JSON Hamiltonian parameters. Example structure:

```json
{
  "QubitsNumber": 10,
  "Hamiltonian": {
    "Type": "SchwingerTemporalGauge",
    "Parameters": {
      "L": 10,
      "a": 0.5,
      "m": 0.1,
      "e0": 0.8
    },
    "Lambda_Charge_Penalty": "Variable"
  },
  "Ansatz": {
    "Type": "EfficientSU2",
    "Reps": 2,
    "Entanglement": "full"
  },
  "Temporal Evolution": {
    "Total_Time": 10.0,
    "Time_Steps": 100,
    "Quench": {
      "Parameters_to_Change": {"e0": 0.5}
    }
  },
  "Backend": {
    "Type": "AerSimulator"
  }
}
```


## Analysis Workflow

### `Results/R01_Persistence.py`

Script to calculate vacuum persistence related results:

1. **Qubit Scaling**: Fit $\Gamma(L)$ to find optimal system size
2. **Best Fit**: Show exponential fit for best $L$ with Schwinger overlay
3. **Regime Visualization**: Plot persistence marking Zeno/Schwinger/revival boundaries
4. **Field Dependence**: Scan $e_0$ values, show $\Gamma$ vs background field
5. **Log Field Dependence**: Log-log plot for Schwinger exponential suppression

Output saved to `../data/` and `../plots/`.

## Running Simulations

### Quick Start

```python
from SchwingerSimulation import SchwingerSimulation
from Utils import loadJsonConfig

# Load pre-configured simulation
config = loadJsonConfig("SchwingerSimulation_v0.json")

# Create and run
simulator = SchwingerSimulation(config)
simulator.run_simulation()

# Results available immediately
print(simulator.evolution_data)
```

### Batch Simulation (Different L)

See `Results/R01_Persistence.py` for the pattern—it sweeps system sizes and saves data locally.

### Custom Configuration

Modify `Configs/SchwingerSimulation_v0.json` or create new config, then instantiate:

```python
import json
with open("MyConfig.json") as f:
    my_config = json.load(f)
sim = SchwingerSimulation(my_config)
sim.run_simulation()
```

## Data Format

### `evolution_data` (DataFrame)

Index: Time (float)  
Columns example (the ones they appear in the config json):
- `Persistence` (float): Vacuum overlap $|\langle\psi(t)|\psi_0\rangle|^2$
- `Energy` (float): Instantaneous energy ⟨E(t)⟩
- `Electric_Field_Avg` (float): Average ⟨E⟩
- `Pair_Creation` (float): $Q(t)$ rate
- `Gauss_Law_Violation` (float): Residual $|Q̂|$

### `initial_state` (Statevector)

Qiskit Statevector object—bare quantum state after VQE.  
Can be exported via `qpy.dump()` for reproducibility.

## Testing

Run unit tests:
```bash
python -m pytest Tests/
```

Or from QuantumSimulation folder:
```python
import Tests.tests_operators as tests
tests.test_hamiltonian_structure()
```

## Future Extensions

- [ ] Tensor Networks Architecture
- [ ] Advanced Physical Results

## Disclaimer

Creation of README.md files has been assisted by AI.

# Lattice-QFT-Quantum-Simulation

Quantum simulation of the **Schwinger effect** in 1+1D lattice QED on NISQ quantum computers using Qiskit.

## Overview

This repository implements the quantum simulation of non-perturbative QED phenomena on gate-based quantum computers. Specifically, it simulates the **Schwinger pair production** (electron-positron pair creation from the vacuum under strong electric fields) in (1+1)-dimensional lattice QED, bridging lattice gauge theory with variational quantum algorithms.

Some results and examples can be viewed as notebooks at QuantumSimulation/Results (only persistance analysis by the moment).

### Key Features

- **Main Implementation**: Schwinger effect simulation in 1+1D lattice QED
- **Variational Quantum Eigensolver (VQE)** for vacuum state preparation
- **Time evolution** using Suzuki-Trotter decomposition
- **Observable calculations**: vacuum persistence, pair creation, Gauss law violations

## Physics Background

### The Schwinger Model

The Schwinger model is the simplest non-trivial gauge theory—1+1D QED with one Dirac fermion coupled to a U(1) gauge field. While soluble classically, it exhibits quintessential QED features:

- **Non-perturbative pair production** from the vacuum
- **Charge confinement** (linearly rising potential between separated charges)
- **Dynamical mass generation**

#### Hamiltonian (Temporal Gauge)

The lattice Hamiltonian simulated in this code, with $e=1$ (natural units), is:

$$H = \sum_n \left[\frac{w}{2}\left(\psi_n^\dagger e^{i\theta_n}\psi_{n+1} + \text{h.c.}\right) + J E_n^2 + m(-1)^n \psi_n^\dagger \psi_n\right]$$

Where:
- $\psi_n$: fermion annihilation operator at site $n$
- $\theta_n = a E_n$: gauge angle (integrated electric field)
- $E_n$: electric field configuration
- $w = 1/(2a)$: hopping amplitude ($a$ = lattice spacing)
- $J = a/2$: gauge coupling
- $m$: fermion mass (staggered term represents relativistic dispersion)

**Parameters**:
- $L$: Number of lattice sites (qubits)
- $a$: Lattice spacing (controls continuum limit)
- $m$: Fermion mass
- $e_0$: Background electric field in the first site of the lattice

### Schwinger Pair Production Formula

In the strong-field regime ($eE \gg m^2$), the exponential decay rate of vacuum persistence is:

$$\Gamma = \int dE \ \mathcal{P}(E), \ \mathcal{P}(E) = \frac{eE}{2\pi} \exp\left(-\frac{\pi m^2}{eE}\right)$$

This non-perturbative formula predicts exponential suppression of the ground state after external field application—the **Schwinger effect**.

## Project Structure

```
└── QuantumSimulation/
    ├── SchwingerSimulation.py       # Main simulation class
    ├── Operators.py                 # Hamiltonian & observable definitions
    ├── circuitBuilder.py            # Quantum circuit construction
    ├── Calculations.py              # Observable calculations
    ├── Plots.py                     # Visualization utilities
    ├── Utils.py                     # Helper functions & I/O
    ├── ResultsAnalysis.py           # Persistence fit & regime analysis
    ├── _config.py                   # Path configuration
    ├── Configs/                     # Hamiltonian parameter JSON configs
    ├── Results/
    │   └── R00_ResultsCommon.py     # Reusable functions for results analysis
    │   └── R01_Persistence.py       # Vacuum persistence analysis
    │   └── R01_Persistence.ipynb    # Vacuum persistence analysis in notebook format
    └── Tests/
        └── tests_operators.py       # Unit tests
        └── tests_simulation.py      # Unit tests
```

Paths to save data are defined in QuantumSimulation/_config.py.
Folders can be modified in that script if needed.

## Workflow: SchwingerSimulation Class

The main `SchwingerSimulation` class orchestrates the full quantum simulation. Understanding its workflow is essential:

### 1. **Initialization**
```python
from QuantumSimulation import SchwingerSimulation
import json

with open("Configs/SchwingerSimulation_v0.json") as f:
    config = json.load(f)

simulator = SchwingerSimulation(config)
```

### 2. **Configuration Structure**

The JSON config specifies all parameters:

```json
{
  "QubitsNumber": 10,
  "Hamiltonian": {
    "Type": "SchwingerTemporalGauge",
    "Parameters": {
      "L": 10,          // Lattice sites
      "a": 0.5,         // Spacing
      "m": 0.1,         // Mass
      "e0": 0.0         // Initial field
    }
  },
  "Ansatz": {
    "Type": "EfficientSU2",
    "Reps": 2
  },
  "Temporal Evolution": {
    "Total_Time": 10.0,
    "Time_Steps": 100,
    "Quench": {
      "Parameters_to_Change": {"e0": 0.5}
    }
  }
}
```

### 3. **Full Simulation Pipeline** (called via `run_simulation()`)

The class internally executes these steps in sequence:

```
┌─────────────────────────────────┐
│ 1. Build Hamiltonian            │  (SchwingerSimulation.get_hamiltonian)
│    ↓ Builds SparsePauliOp from  │
│      config parameters          │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ 2. Check Charge Symmetry (U(1)) │  (Operators.checkChargeSymmetry)
│    ↓ Validate Gauss law         │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ 3. Add Penalty Term (optional)  │  (Ensure Gauss law constraint)
│    ↓ H → H + λ·Q̂²               │
└──────────────┬──────────────────┘
               ↓
┌──────────────────────────────────────────────┐
│ 4. Build Ansatz Circuit                      │  (SchwingerSimulation.get_ansatz)
│    ↓ EfficientSU2, ExcitationPreserving, ... │
└──────────────┬───────────────────────────────┘
               ↓
┌─────────────────────────────────┐
│ 5. VQE: Optimize Vacuum State   │  (SchwingerSimulation.get_vacuum)
│    ↓ Minimize ⟨ψ|H|ψ⟩            │
│    ↓ Return |ψ_vac⟩ & E_0       │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ 6. Temporal Evolution (optional)│  (SchwingerSimulation.evolve_state)
│    ↓ Apply time-dependent H(t)  │
│    ↓ Quench params. Ex: m, e0   │
│    ↓ Quench: H₀(e0=0) → H(e0)   │
│    ↓ Evolve |ψ_vac⟩ for time T  │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ 7. Calculate Observables        │  (Calculations.*)
│    ↓ Persistence ⟨ψ|ψ₀⟩²         │
│    ↓ Pair creation Q̂            │
│    ↓ Electric field ⟨Ê⟩          │
└─────────────────────────────────┘
```

All data is stored in:
- `simulator.evolution_data` (pandas DataFrame with time series)
- `simulator.initial_state` (optimized vacuum Statevector)
- `simulator.vacuum_energy` (ground state energy)

### 4. **Imports Used Inside the Class**

Key imports and their roles:

```python
from circuitBuilder import buildCircuit, addGate               # Circuit construction
from Operators import buildSchwingerHamiltonianTemporalGauge  # Hamiltonian
from Calculations import calculateEnergy, calculateVacuumPersistence  # Observables
from qiskit_aer import AerSimulator                             # Statevector sim
from scipy.optimize import minimize                             # VQE optimizer
```

## Usage Example

### Basic Simulation

```python
from QuantumSimulation import SchwingerSimulation
import json

# Load configuration
with open("QuantumSimulation/Configs/SchwingerSimulation_v0.json") as f:
    config = json.load(f)

# Create simulator
simulator = SchwingerSimulation(config)

# Run full pipeline: VQE + time evolution
simulator.run_simulation()

# Access results
print(f"Ground state energy: {simulator.vacuum_energy:.4f}")
print(f"Evolution data shape: {simulator.evolution_data.shape}")

# Plot persistence decay
import matplotlib.pyplot as plt
plt.plot(simulator.evolution_data.index, simulator.evolution_data["Persistence"])
plt.xlabel("Time"); plt.ylabel("Vacuum Persistence")
plt.show()
```

### Analysis: Fit Persistence to Schwinger Formula

```python
from QuantumSimulation.ResultsAnalysis import fit_persistence

# Fit exponential decay and compare with Schwinger prediction
gamma_sim, gamma_analytical, gamma_err, eE, cut_offs = fit_persistence(
    simulator.evolution_data, 
    config, 
    initial_state=simulator.initial_state,
    use_offset=False
)

print(f"Simulated decay rate Γ: {gamma_sim:.4f} ± {gamma_err:.4f}")
print(f"Schwinger predicts Γ: {gamma_analytical:.4f}")
print(f"Relative error: {abs(gamma_sim - gamma_analytical)/gamma_analytical * 100:.1f}%")
print(f"Regime boundaries: {cut_offs}")
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/oscar-vbl/Lattice-QFT-Quantum-Simulation.git
cd Lattice-QFT-Quantum-Simulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install in development mode:
```bash
pip install -e .
```

## Current Limitations

- **Only Schwinger Model Implemented**: This is the sole gauge theory currently supported. Extensions to SU(2) or higher-dimensional QED are future work.
- **Small System Sizes**: Simulations limited to ~10-20 qubits on current NISQ devices due to noise and circuit depth.
- **Classical Density Matrix Only**: Uses Qiskit Aer statevector simulator; no noise models implemented yet.
- **Vacuum Perp to Ground State**: Strong-field regimes require careful initialization and choice of ansatz.

## Output

Results include:

- **evolution_data.csv**: Time-series observables (Persistence, Electric field, Pair creation)
- **Figures**: Persistence decay curves, $\Gamma$ vs system size, regime boundaries
- **Analysis**: Best-fit Schwinger rate, deviation from theory, cut-off times

## References

- Qiskit Documentation: https://qiskit.org/documentation/

## Disclaimer

Creation of README.md files has been assisted by AI.


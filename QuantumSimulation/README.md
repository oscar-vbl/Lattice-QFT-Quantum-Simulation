# QuantumSimulation

Technical documentation for the core quantum-simulation package.

`QuantumSimulation` provides reusable components for constructing lattice Hamiltonians, preparing quantum states, executing real-time dynamics and evaluating physical observables. The current reference model is the \(1+1\)-dimensional Schwinger model, while the internal architecture is intended to support broader lattice-field-theory and quantum many-body applications.

## Design Principles

The package follows several guiding principles:

- **Modularity:** Hamiltonians, ansatzes, execution engines and observables are separated where practical.
- **Reproducibility:** Physical and algorithmic parameters are supplied through configuration files.
- **Extensibility:** New observables and circuit families should be addable without rewriting the complete workflow.
- **Validation:** Exact calculations, symmetry checks and automated tests provide reference points for approximate algorithms.
- **Backend awareness:** Ideal, simulator-based and hardware-oriented execution paths are treated as distinct use cases.
- **Research usability:** Stored states, tabular outputs and analysis scripts support repeatable parameter studies.

## Package Layout

```text
QuantumSimulation/
├── Tests/                     # Unit and integration tests
├── core/                      # Core execution and evolution infrastructure
├── observables/               # Modular observable implementations
├── Ansatzes.py                # Variational circuit families
├── Calculations.py            # Calculation helpers and compatibility layer
├── Operators.py               # Hamiltonians, charges and field operators
├── Plots.py                   # Plotting functions
├── ResultsAnalysis.py         # Fitting and physical post-processing
├── SchwingerSimulation.py     # High-level simulation interface
├── Utils.py                   # I/O, configuration and helper utilities
├── __init__.py
├── _config.py                 # Internal project paths
└── circuitBuilder.py          # Circuit-construction helpers
```

## High-Level Workflow

The main simulation pipeline is:

```text
Configuration
    ↓
Hamiltonian and constraints
    ↓
Ansatz construction
    ↓
Vacuum or initial-state preparation
    ↓
Quench / real-time evolution
    ↓
Observable evaluation
    ↓
Uncertainty and post-processing
    ↓
DataFrames, saved states and figures
```

The public entry point is the `SchwingerSimulation` class.

```python
from QuantumSimulation import SchwingerSimulation
from QuantumSimulation.Utils import loadJsonConfig

config = loadJsonConfig("configs/example.json")
simulator = SchwingerSimulation(config)
simulator.run_simulation()
```

## `SchwingerSimulation.py`

`SchwingerSimulation` is the high-level orchestration class. It coordinates the configured simulation without requiring analysis notebooks to reproduce low-level setup logic.

Principal responsibilities include:

- reading and validating configuration data;
- constructing the preparation and evolution Hamiltonians;
- creating a variational ansatz;
- optimizing or loading an initial state;
- configuring quantum primitives or direct numerical execution;
- applying quenches and real-time evolution;
- evaluating selected observables;
- collecting values and uncertainties in structured outputs.

Typical public attributes after execution include:

```python
simulator.initial_state
simulator.vacuum_energy
simulator.evolution_data
simulator.evolution_error
simulator.all_trotter_evolution_data
simulator.all_trotter_evolution_error
```

The exact set of populated attributes depends on the selected workflow and configuration.

## `core/`

The `core` package contains lower-level execution infrastructure extracted from the high-level simulation class. Its purpose is to keep backend setup, evolution strategies and reusable numerical logic independent from the physical analysis notebooks.

Depending on the active implementation, core functionality may include:

- execution-engine configuration;
- primitive setup;
- incremental state evolution;
- Trotter-step construction;
- result collection;
- shared simulation data structures.

This package is expected to evolve as execution paths are further separated into direct-statevector, Aer and hardware-oriented engines.

## `observables/`

The observable framework defines quantities evaluated during state preparation or temporal evolution.

Current observable families include:

- energy;
- vacuum persistence;
- pair creation;
- electric field;
- Gauss-law violation.

An observable may implement different evaluation routes:

- exact calculation from a `Statevector`;
- Estimator-based expectation values;
- Sampler-based measurement circuits;
- post-processing of grouped primitive results;
- uncertainty propagation.

A simplified conceptual interface is:

```python
class Observable:
    name: str
    primitive_type: str

    def get_operators(self):
        ...

    def get_pub(self, circuit):
        ...

    def calculate_exact(self, state):
        ...

    def process_result_data(self, values, errors):
        ...
```

New observables should keep physical operator construction separate from plotting and high-level interpretation.

## `Operators.py`

`Operators.py` defines model Hamiltonians and physical operators in qubit form.

The current implementation includes:

- the Schwinger Hamiltonian in temporal gauge;
- staggered charge operators;
- total-charge operators;
- electric-field operators derived from Gauss's law;
- pair-number operators;
- Gauss-law generators.

Operator conventions, lattice indexing and boundary conditions should be documented explicitly whenever new models are added.

## `Ansatzes.py`

`Ansatzes.py` contains variational circuit families used for state preparation and benchmarking.

Supported or explored ansatz categories include:

- hardware-efficient circuits;
- excitation-preserving circuits;
- Hamiltonian variational ansatzes;
- custom physics-informed constructions.

The ansatz-validation workflows compare candidates through quantities such as:

- ground-state energy error;
- state fidelity;
- convergence behaviour;
- optimization time;
- transpiled depth;
- entangling-gate count;
- gradient-variance or barren-plateau indicators.

The goal is not to define a universally optimal circuit, but to provide reproducible criteria for selecting an ansatz for a given physical regime and execution target.

## `circuitBuilder.py`

This module contains reusable utilities for constructing quantum circuits from configuration data.

Responsibilities may include:

- initial-state preparation;
- parameterized layers;
- entanglement patterns;
- supported Qiskit circuit-library ansatzes;
- custom gate insertion;
- circuit composition.

## `Calculations.py`

`Calculations.py` contains shared numerical helpers and compatibility functions. As the observable-oriented architecture develops, new code should generally use the classes under `observables/` rather than legacy `calculate...` functions.

The compatibility layer may remain available temporarily so earlier notebooks and scripts continue to run during migration.

## `ResultsAnalysis.py`

`ResultsAnalysis.py` provides physical post-processing that is intentionally kept outside the simulation engine.

Current analysis capabilities include:

- exponential persistence fitting;
- temporal-window selection;
- vacuum-decay-rate extraction;
- comparison with analytical estimates;
- finite-size and parameter scans;
- uncertainty-aware regression;
- regime diagnostics.

Future analysis modules may cover excited-state spectroscopy, topological sectors, entanglement scaling and fidelity-based phase diagnostics.

## `Plots.py`

`Plots.py` contains reusable visualization functions for simulation and analysis outputs.

Examples include:

- ansatz comparisons;
- energy and fidelity convergence;
- circuit-resource plots;
- persistence curves and temporal regimes;
- decay-rate scaling;
- field-dependence studies;
- uncertainty bands and analytical comparisons.

Plotting functions should receive processed data rather than reproduce physical calculations internally.

## `Utils.py`

General-purpose utilities include:

- configuration loading;
- DataFrame and state persistence;
- simulator-instance loading;
- path handling;
- logging and timing;
- formatting helpers.

## `_config.py`

Internal project paths are centralized here so notebooks and scripts do not depend on machine-specific absolute paths.

## Configuration Model

Simulations are controlled through JSON files under the repository-level `configs/` directory.

A simplified example is:

```json
{
  "QubitsNumber": 10,
  "Hamiltonian": {
    "Type": "SchwingerTemporalGauge",
    "Parameters": {
      "L": 10,
      "a": 0.5,
      "m": 0.1,
      "e0": 0.0
    }
  },
  "Ansatz": {
    "Type": "ExcitationPreserving",
    "Reps": 2
  },
  "Temporal Evolution": {
    "Total_Time": 5.0,
    "Time_Steps": 100,
    "Quench": {
      "Parameters_to_Change": {
        "e0": 0.5
      }
    },
    "Observables": {
      "Observables_List": [
        "Persistence",
        "Energy",
        "Electric_Field"
      ]
    }
  },
  "Backend": {
    "Type": "Aer",
    "Options": {
      "Shots": 1024
    }
  }
}
```

The exact schema may evolve. Use the current files under `configs/` as executable references.

## Execution Modes

The package supports, or is being organized around, several execution strategies:

### Direct numerical execution

Suitable for ideal validation and small systems. States are evolved directly and observables can be evaluated exactly.

### Aer simulation

Suitable for optimized local simulation, finite-shot studies, optional noise models and backend-like execution.

### Primitive-oriented execution

Suitable for Qiskit Estimator/Sampler workflows and future hardware execution. Estimator observables are grouped by state where possible to reduce redundant primitive calls.

These execution strategies should produce compatible high-level outputs even when their internal implementations differ.

## Temporal Evolution and Trotter Studies

Real-time dynamics are generated from configurable Suzuki-Trotter decompositions. Multiple temporal resolutions can be evaluated in the same workflow to study discretization effects.

Representative outputs are organized by Trotter key:

```python
simulator.all_trotter_evolution_data["dt"]
simulator.all_trotter_evolution_data["dt_half"]
```

Associated error structures are stored separately where available.

Richardson-style extrapolation can be used in downstream analysis to estimate or reduce leading Trotter errors, provided the assumed error scaling is valid in the selected regime.

## Research Notebooks

The repository-level `results/` directory contains reproducible research workflows built on top of this package.

Representative studies include:

### Ansatz validation

- energy and fidelity accuracy;
- optimizer convergence;
- circuit depth and entangling-gate counts;
- system-size scaling;
- gradient-variance diagnostics.

### Vacuum persistence

- finite-size scans;
- temporal-regime identification;
- uncertainty-aware exponential fits;
- effective vacuum-decay rates;
- field-strength dependence;
- Schwinger-like non-perturbative scaling.

These workflows illustrate the package API. They do not limit the physical scope of future studies.

## Data Products

Simulation outputs are commonly exposed as pandas DataFrames indexed by physical time.

Example:

```python
simulator.evolution_data.head()
simulator.evolution_error.head()
```

Possible columns include:

```text
Persistence
Energy
Electric_Field_0
Electric_Field_1
Pair_Creation_Electrons
Pair_Creation_Positrons
Gauss_Law_Violation
```

Column availability depends on the configured observables.

## Testing

From the repository root:

```bash
python -m pytest QuantumSimulation/Tests
```

The test suite covers, as applicable:

- Hamiltonian structure and Hermiticity;
- charge symmetry;
- gauge-related operators;
- ansatz and circuit construction;
- observable calculations;
- end-to-end simulation workflows;
- multiple Trotter configurations;
- algorithmic-mitigation paths.

## Development Guidelines

When extending the package:

1. keep physical conventions explicit;
2. add tests for new Hamiltonians or observables;
3. avoid embedding analysis-specific logic in the simulation engine;
4. expose results through stable data structures;
5. preserve compatibility with configuration-driven workflows;
6. document assumptions about boundaries, gauge choice and normalization;
7. validate approximate methods against exact small-system references.

## Planned Extensions

Potential extensions include:

- excited-state solvers and spectral gaps;
- vacuum topology and symmetry-sector studies;
- entanglement entropy and mutual information;
- fidelity susceptibility and geometric phases;
- time-dependent field protocols;
- tensor-network-inspired methods;
- additional lattice gauge theories;
- noise-aware and hardware-executed simulations.

## Scope

The purpose of `QuantumSimulation` is not only to reproduce one Schwinger-effect calculation. The package is intended as a reusable research environment for developing, validating and comparing quantum-simulation methods for lattice quantum field theories and related many-body systems.

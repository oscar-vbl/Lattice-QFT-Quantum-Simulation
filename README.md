# Lattice-QFT-Quantum-Simulation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-Quantum%20Simulation-6929C4.svg)](https://www.ibm.com/quantum/qiskit)
[![Tests](https://github.com/oscar-vbl/Lattice-QFT-Quantum-Simulation/actions/workflows/tests.yml/badge.svg)](https://github.com/oscar-vbl/Lattice-QFT-Quantum-Simulation/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21787494.svg)](https://doi.org/10.5281/zenodo.21787494)

A modular Python framework for studying lattice quantum field theories with variational algorithms, quantum circuits and classical reference methods.

The current implementation focuses on the lattice Schwinger model, quantum electrodynamics in \(1+1\) dimensions. It provides a compact testbed for vacuum preparation, gauge constraints, particle production, background-field quenches and real-time quantum dynamics.

The project combines reusable simulation software with reproducible research workflows. Its architecture separates physical models, operators, observables, algorithms, execution backends, temporal evolution, mitigation and result processing.

> This repository contains research software under active development. Interfaces, configuration formats and numerical workflows may evolve before a stable release.

## Overview

Lattice formulations map quantum field theories onto finite-dimensional systems that can be represented with qubits. This makes it possible to investigate selected static and dynamical phenomena using gate-based quantum algorithms, while retaining exact and classical simulation methods as validation references.

The repository provides infrastructure for:

- constructing lattice Hamiltonians and physical operators;
- preparing vacuum and low-energy states with variational algorithms;
- comparing hardware-efficient and physics-informed ansatzes;
- validating symmetries and physical constraints;
- simulating real-time evolution and parameter quenches;
- measuring modular physical observables;
- comparing exact, ideal, noisy and primitive-based execution;
- analysing statistical and algorithmic uncertainties;
- mitigating Trotter discretization errors;
- building reproducible numerical studies and research notebooks.

The software is currently specialized in the Schwinger model. The modular design is intended to support additional observables, algorithms and lattice models as the project develops.

## Physical model

The reference implementation uses a lattice formulation of the massive Schwinger model in temporal gauge with staggered fermions and open boundary conditions.

The implemented workflows include:

- staggered-fermion Hamiltonian construction;
- hopping, mass and electric-field contributions;
- total-charge operators;
- electric-field operators;
- particle-number observables;
- gauge-consistency diagnostics;
- variational vacuum preparation;
- background-field quenches;
- Suzuki-Trotter real-time evolution;
- exact matrix evolution for small systems;
- vacuum-persistence measurements;
- Richardson extrapolation of Trotterized observables.

The accessible system sizes remain limited by statevector memory, circuit depth, optimization cost and current quantum-hardware noise. The framework is therefore primarily intended for small-system studies, method validation and reproducible quantum-simulation experiments.

## Main features

### Modular physical representation

Hamiltonians and physical quantities are separated into reusable components:

- physical parameter dataclasses;
- Hamiltonian builders;
- symbolic operator definitions;
- conversion to Qiskit operators;
- modular observable implementations;
- exact and primitive-based evaluation paths.

This separation allows the same physical quantity to be used during state validation, temporal evolution and post-processing without embedding its implementation in the main simulation class.

### Variational state preparation

The package supports vacuum preparation through variational quantum eigensolver workflows, including:

- standard parameterized ansatzes;
- excitation-preserving circuits;
- Hamiltonian variational ansatzes;
- configurable classical optimizers;
- explicit or generated initial parameters;
- optional gradient-based optimization;
- exact and primitive-based energy evaluation;
- optimization-history tracking.

The research workflows compare ansatz families using energy, fidelity, circuit depth, entangling-gate counts and optimization behaviour.

### Temporal evolution

Real-time dynamics are handled through a dedicated dynamics layer supporting:

- parameter quenches;
- exact matrix evolution for small systems;
- Pauli evolution gates;
- Suzuki-Trotter product formulas;
- multiple Trotter step sizes;
- exact and primitive-based observable evaluation;
- configurable temporal observables;
- structured trajectory results;
- execution timing and workload metadata.

Evolution, mitigation and result processing are implemented as separate stages.

### Algorithmic mitigation

The mitigation layer currently includes Richardson extrapolation for comparing compatible Trotter trajectories.

Values and statistical uncertainties are processed separately. Richardson extrapolation corrects observable estimates, while uncertainty propagation accounts for the linear extrapolation weights.

A mitigated observable trajectory does not represent a mitigated quantum state. When a final state is required, the state from the finest underlying physical trajectory is retained.

### Execution backends

The execution layer supports different simulation contexts through a common primitive bundle:

- exact calculations without primitives;
- Qiskit statevector primitives;
- Aer simulation;
- configurable Aer noise models;
- IBM Runtime integration for explicitly selected workflows.

IBM Runtime tests and remote execution are disabled in the default continuous-integration workflow.

### Physical observables

The current observable set includes:

- energy;
- total charge;
- vacuum persistence;
- electron and positron occupation;
- electric field;
- Gauss-law consistency.

The observable interface is extensible and supports scalar and multi-component results with associated uncertainties.

### Diagnostics and validation

The package includes infrastructure for:

- Hamiltonian hermiticity checks;
- conserved-charge checks;
- symmetry commutators;
- state-sector validation;
- exact-statevector references;
- gauge-consistency diagnostics;
- comparison between exact and primitive-based measurements.

### Testing and continuous integration

The test suite includes unit and integration tests covering:

- core configuration and result types;
- Hamiltonians and physical operators;
- exact observable evaluation;
- Qiskit primitives;
- ansatz construction;
- VQE execution;
- dynamics configuration;
- Richardson extrapolation;
- hardware and noise builders;
- diagnostics;
- high-level simulation orchestration.

Tests are executed automatically through GitHub Actions on supported Python versions. IBM Runtime tests are retained as an optional category and excluded from the default workflow.

## Repository structure

```text
Lattice-QFT-Quantum-Simulation/
├── .github/
│   └── workflows/              # Continuous-integration workflows
├── QuantumSimulation/
│   ├── algorithms/             # Variational and quantum algorithms
│   ├── ansatz/                 # Initial states and ansatz families
│   ├── circuits/               # Circuit construction and transpilation
│   ├── core/
│   │   └── types/              # Typed configuration and result contracts
│   ├── diagnostics/            # Physical validation and symmetry checks
│   ├── dynamics/               # Real-time evolution
│   ├── hamiltonians/           # Physical Hamiltonian models
│   ├── hardware/               # Backends, primitives and noise models
│   ├── mitigation/             # Algorithmic error mitigation
│   ├── observables/            # Physical observable implementations
│   ├── operators/              # Symbolic physical operators
│   ├── processing/             # Trajectory and tabular processing
│   ├── SchwingerSimulation.py  # High-level simulation interface
│   ├── Utils.py                # I/O and general utilities
│   └── README.md               # Package-level technical documentation
├── configs/                    # Reproducible simulation configurations
├── results/                    # Research notebooks and study-specific assets
├── tests/
│   ├── unit/                   # Unit tests
│   └── integration/            # End-to-end and multi-module tests
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

### Clone the repository

```bash
git clone https://github.com/oscar-vbl/Lattice-QFT-Quantum-Simulation.git
cd Lattice-QFT-Quantum-Simulation
```

### Create an isolated environment

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies and the package

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

For development and testing:

```bash
python -m pip install -e ".[test]"
```

## Quick start

A simulation is configured through a Python dictionary or a JSON file.

```python
from QuantumSimulation.SchwingerSimulation import (
    SchwingerSimulation,
)
from QuantumSimulation.Utils import loadJsonConfig


config = loadJsonConfig(
    "configs/example.json"
)

simulation = SchwingerSimulation(config)
simulation.run_simulation()

print(simulation.vacuum_energy)

if simulation.evolution_config is not None:
    print(simulation.evolution_data.head())
```

A typical workflow consists of:

1. defining physical and numerical parameters;
2. constructing the Hamiltonian;
3. applying configured symmetry constraints;
4. preparing or loading an initial state;
5. validating the prepared state;
6. applying an optional quench;
7. executing exact or Trotterized temporal evolution;
8. measuring the selected observables;
9. applying configured mitigation;
10. processing and analysing the resulting trajectories.

See the configuration files and research notebooks for complete executable examples.

## Configuration

The configuration system controls:

- lattice size and physical parameters;
- Hamiltonian model and gauge;
- backend and primitive selection;
- shot count and target precision;
- ansatz family and depth;
- optimizer settings;
- symmetry and state validation;
- temporal evolution;
- parameter quenches;
- Trotter variants;
- observable selection;
- mitigation settings.

Configuration is parsed into typed dataclasses before execution. This centralizes validation and prevents individual modules from depending directly on the raw JSON structure.

The package-level documentation in `QuantumSimulation/README.md` describes the available fields in greater detail.

## Research workflows

The `results/` directory contains reproducible studies built on top of the simulation package.

Current workflows include:

### Ansatz validation

Comparison of variational circuit families through:

- ground-state energy;
- state fidelity;
- optimization behaviour;
- circuit depth;
- two-qubit gate counts;
- parameter count;
- gradient statistics.

### Vacuum persistence and pair production

Study of real-time dynamics following a background-field quench, including:

- vacuum-persistence probability;
- electron and positron occupation;
- electric-field evolution;
- finite-size comparisons;
- effective vacuum-decay rates;
- Trotter convergence;
- uncertainty propagation.

The notebooks provide research examples rather than defining the complete scope of the package.

## Testing

Run the default test suite from the repository root:

```bash
python -m pytest \
    -m "not slow and not ibm"
```

Run tests with coverage:

```bash
python -m pytest \
    -m "not slow and not ibm" \
    --cov=QuantumSimulation \
    --cov-report=term-missing \
    --cov-report=html
```

The HTML coverage report is written to:

```text
htmlcov/index.html
```

Tests requiring IBM Runtime are marked separately and are not executed by default.

## Reproducibility

For reproducible numerical studies, record:

- repository release or commit;
- configuration file;
- Python and dependency versions;
- random seeds;
- backend and primitive configuration;
- shot count and precision;
- ansatz and optimizer settings;
- initial variational parameters;
- quench parameters;
- Trotter synthesis and step sizes;
- mitigation configuration.

Exact calculations, finite-shot simulations and hardware executions should be distinguished clearly when interpreting results.

## Scope and limitations

The current implementation is intended for research, validation and small-system simulation.

Relevant limitations include:

- the physical implementation currently focuses on the Schwinger model;
- the gauge-eliminated formulation uses open boundary conditions;
- exact statevector and matrix methods scale exponentially;
- VQE results depend on the ansatz, optimizer and initialization;
- Trotter errors and hardware errors are distinct sources of uncertainty;
- Richardson extrapolation mitigates compatible discretization errors but does not produce a corrected quantum state;
- current hardware limits restrict accessible circuit depth and lattice size;
- continuum and thermodynamic conclusions require dedicated scaling analyses.

No ensions include:

- vacuum topology and background-angle dependence;
- phase-transition diagnostics;
- fidelity susceptibility and geometric phases;
- excited-state algorithms such as VQD;
- entanglement entropy and mutual information;
- time-dependent external fields;
- additional product formulas and evolution strategies;
- readout and hardware-noise mitigation;
- symmetry verification and postselection;
- additional Abelian and non-Abelian lattice gauge models;
- scalable execution on quantum hardware and high-performance simulators.

The roadmap is exploratory. New functionality is introduced when it is physically motivated and can be validated against established references.

## Citation

If this software contributes to academic work, cite the specific release or commit used in the study.

Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

```bibtex
@software{vicente_blazquez_lattice_qft_quantum_simulation,
  author  = {Vicente Blázquez, Oscar},
  title   = {Lattice-QFT-Quantum-Simulation},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.21787494},
  url     = {https://doi.org/10.5281/zenodo.21787494}
}
```

## License

This project is distributed under the terms of the [MIT License](LICENSE).

## Disclaimer

This repository contains research software under active development. Numerical and physical results should be interpreted together with the model assumptions, finite-size effects, discretization choices, optimization behaviour and uncertainty analyses documented in the corresponding workflows.
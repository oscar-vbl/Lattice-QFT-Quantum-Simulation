# Lattice-QFT-Quantum-Simulation

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.x-6929C4)](https://www.ibm.com/quantum/qiskit)
[![Tests](https://github.com/oscar-vbl/Lattice-QFT-Quantum-Simulation/actions/workflows/tests.yml/badge.svg)](https://github.com/oscar-vbl/Lattice-QFT-Quantum-Simulation/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1103332742.svg)](https://doi.org/10.5281/zenodo.21787494)

A modular research framework for the quantum simulation of lattice quantum field theories and gauge models using variational and circuit-based quantum algorithms.

The current reference implementation focuses on the **Schwinger model**, quantum electrodynamics in \(1+1\) dimensions. The model provides a compact setting for studying non-perturbative field-theory phenomena, gauge constraints, vacuum structure and real-time dynamics on gate-based quantum computers. The software architecture is intended to evolve beyond this first use case through additional models, observables and simulation methods.

> [!NOTE]
> This repository is under active research development. Interfaces, configuration schemas and analysis workflows may evolve before a stable release.

## Overview

Quantum simulation offers a route for investigating strongly correlated systems and real-time quantum dynamics that can become difficult for conventional classical approaches. Lattice formulations provide a finite-dimensional representation in which field-theory Hamiltonians can be mapped to qubits and studied with quantum algorithms.

This repository provides reusable infrastructure for:

- constructing lattice Hamiltonians and symmetry operators;
- preparing vacuum and low-energy states with variational algorithms;
- comparing hardware-efficient and physics-informed ansatzes;
- simulating real-time evolution and parameter quenches;
- evaluating modular physical observables;
- estimating statistical and algorithmic uncertainties;
- benchmarking quantum circuits and optimization workflows;
- producing reproducible numerical studies and research notebooks.

## Current Reference Model

The framework currently implements the lattice Schwinger model in temporal gauge, including workflows for:

- staggered-fermion Hamiltonian construction;
- charge-sector and Gauss-law diagnostics;
- variational vacuum preparation;
- Suzuki-Trotter real-time evolution;
- background-field quenches;
- vacuum-persistence probability;
- pair-creation observables;
- electric-field observables;
- finite-size and field-strength scans;
- Trotter-error studies and Richardson-style extrapolation.

The Schwinger model is treated as the first research application of the framework, not as its final scope.

## Main Features

### Configurable simulation pipeline

- JSON-driven physical, variational and execution parameters
- Exact statevector, Aer and primitive-oriented workflows
- Reusable initial states and saved simulation data
- Multiple Trotter discretizations within the same experiment
- Pluggable quench parameters and observable selections

### Variational state preparation

- Variational Quantum Eigensolver workflows
- Hardware-efficient ansatzes
- Excitation-preserving circuits
- Hamiltonian variational ansatzes
- Ansatz benchmarking through energy, fidelity and circuit resources
- Optimization-landscape and barren-plateau diagnostics

### Modular observables

- Energy
- Vacuum persistence
- Pair creation
- Electric field
- Gauss-law violation

The observable interface is designed so additional physical and quantum-information quantities can be introduced without rewriting the simulation engine.

### Validation and uncertainty analysis

- Charge-symmetry checks
- Gauge-constraint diagnostics
- Exact-statevector references
- Statistical measurement uncertainty
- Trotter-discretization comparisons
- Fit and finite-window uncertainties in downstream analyses
- Automated tests through GitHub Actions

## Repository Structure

```text
Lattice-QFT-Quantum-Simulation/
├── .github/
│   └── workflows/                 # Continuous integration
├── OtherScripts/                  # Auxiliary or exploratory scripts
├── QuantumSimulation/
│   ├── Tests/                     # Unit and integration tests
│   ├── core/                      # Core execution infrastructure
│   ├── observables/               # Observable implementations
│   ├── Ansatzes.py                # Variational circuit families
│   ├── Calculations.py            # Compatibility and calculation utilities
│   ├── Operators.py               # Hamiltonians and physical operators
│   ├── Plots.py                   # Visualization utilities
│   ├── README.md                  # Technical package documentation
│   ├── ResultsAnalysis.py         # Post-processing and fitting
│   ├── SchwingerSimulation.py     # High-level simulation interface
│   ├── Utils.py                   # I/O and general helpers
│   ├── __init__.py
│   ├── _config.py                 # Internal path configuration
│   └── circuitBuilder.py          # Circuit-construction utilities
├── configs/                       # Reproducible simulation configurations
├── results/                       # Research notebooks and analysis workflows
├── .gitattributes
├── .gitignore
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

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Install the package

```bash
python -m pip install --upgrade pip
pip install -e .
```

Alternatively, install the declared dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from QuantumSimulation import SchwingerSimulation
from QuantumSimulation.Utils import loadJsonConfig

config = loadJsonConfig("configs/example.json")
simulator = SchwingerSimulation(config)
simulator.run_simulation()

print(simulator.vacuum_energy)
print(simulator.evolution_data.head())
```

A typical workflow consists of:

1. defining the model and numerical parameters in a configuration file;
2. constructing the Hamiltonian and selected symmetry constraints;
3. preparing the initial state variationally or loading a stored state;
4. applying a quench or real-time evolution protocol;
5. evaluating the configured observables;
6. analysing uncertainties, scaling and physical regimes.

Configuration filenames evolve with the research workflows. See [`configs/`](configs/) and the notebooks under [`results/`](results/) for current executable examples.

## Research Workflows

The `results/` directory contains reproducible notebooks and supporting scripts built on top of the simulation package. The workflows are examples of how the framework can be used, rather than a complete definition of its scope.

Current studies include:

- **Ansatz validation:** comparison of variational families through energy, fidelity, optimization behaviour, circuit depth and entangling-gate counts.
- **Vacuum persistence:** finite-size analysis, extraction of an effective vacuum-decay rate, uncertainty propagation and comparison with Schwinger-like behaviour.

Planned studies include vacuum topology, excited-state spectroscopy, entanglement diagnostics and additional real-time protocols.

## Testing

Run the complete test suite from the repository root:

```bash
python -m pytest
```

Or run only the package tests:

```bash
python -m pytest QuantumSimulation/Tests
```

The GitHub Actions workflow runs automated checks on supported Python environments.

## Roadmap

Possible research and software extensions include:

- excited-state algorithms, including VQD-like workflows;
- entanglement entropy and mutual information;
- fidelity susceptibility and geometric diagnostics;
- topological sectors and discrete-symmetry studies;
- time-dependent external fields and pulse protocols;
- tensor-network-inspired ansatzes;
- measurement and hardware-noise mitigation;
- additional Abelian and non-Abelian lattice gauge models;
- scalable execution on quantum hardware and high-performance simulators.

The roadmap is exploratory. New studies are introduced when they are physically motivated and compatible with the validated simulation infrastructure.

## Reproducibility

Simulation parameters are stored in configuration files, while generated states, tabular data and plots are handled through the repository utilities. For reproducible academic use, record:

- the repository release or commit;
- the configuration file;
- the random seed, when applicable;
- the backend and primitive versions;
- the ansatz and optimizer settings;
- the Trotter discretization and measurement budget.

## Citation

If this software contributes to academic work, please cite the specific repository release used in the study. Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).

```bibtex
@software{vicente_blazquez_lattice_qft_quantum_simulation,
  author  = {Vicente Blázquez, Oscar},
  title   = {{Lattice-QFT-Quantum-Simulation}},
  year    = {2026},
  note    = {Quantum simulation framework for lattice quantum field theories},
  version = {0.1.0},
  doi     = {10.5281/zenodo.21787494},
  url     = {https://doi.org/10.5281/zenodo.21787494}
}
```

## License

This project is distributed under the terms of the [MIT License](LICENSE).

## Disclaimer

This is research software under active development. Numerical and physical results should be interpreted together with the assumptions, finite-size effects, discretization choices and uncertainty analyses documented in the corresponding workflows.

Parts of the documentation were initially drafted with AI assistance and subsequently reviewed and edited by the repository author.

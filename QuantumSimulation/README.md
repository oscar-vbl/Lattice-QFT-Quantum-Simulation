# QuantumSimulation

Technical documentation for the core simulation package of the
`Lattice-QFT-Quantum-Simulation` repository.

`QuantumSimulation` provides modular components for constructing lattice
Hamiltonians, preparing quantum states, performing real-time evolution,
measuring physical observables and processing simulation results.

The current physical implementation focuses on the massive Schwinger model
in \(1+1\) dimensions. The package is designed for small-system quantum
simulation, algorithm validation and reproducible numerical experiments
using exact methods, Qiskit primitives, Aer simulators and optional IBM
Runtime execution.

> The package is research software under active development. Public
> interfaces and configuration fields may evolve before a stable release.

## Package responsibilities

The package covers the complete simulation pipeline:

```text
Configuration
    |
    v
Hamiltonian and physical operators
    |
    v
Symmetry validation and penalties
    |
    v
Initial state and variational ansatz
    |
    v
Vacuum preparation or supplied state
    |
    v
Ground-state diagnostics
    |
    v
Parameter quench
    |
    v
Exact or Trotterized temporal evolution
    |
    v
Observable measurement
    |
    v
Algorithmic mitigation
    |
    v
Processed trajectories and DataFrames
```

The high-level public interface is `SchwingerSimulation`. Lower-level
modules can also be used independently when more control is required.

## Design principles

The implementation follows several principles.

### Separation of physical and execution layers

Physical models and operators are defined independently from the backend
used to evaluate them.

A physical operator is first represented in the logical lattice model and
then converted to the corresponding Qiskit representation for execution.

### Configuration-driven execution

Physical, algorithmic and backend options are supplied through configuration
files and parsed into typed dataclasses before execution.

Internal modules should consume typed configuration objects rather than
reading directly from the raw JSON document.

### Modular observables

Every observable defines:

- the physical operator or measurement circuit that it requires;
- the primitive used for its evaluation;
- exact-state evaluation when available;
- primitive-result decoding;
- values and uncertainty processing.

Observable definitions do not contain plotting or notebook-specific
analysis.

### Independent dynamics, mitigation and processing

Temporal evolution, algorithmic mitigation and output formatting are
separate operations.

- Dynamics generates physical trajectories.
- Measurement evaluates observables along those trajectories.
- Mitigation creates corrected observable trajectories.
- Processing converts structured results into analysis-friendly formats.

A mitigated observable trajectory does not necessarily correspond to a
mitigated quantum state.

### Exact small-system references

Exact statevector and matrix calculations provide reference results for
validating variational, Trotterized and finite-shot execution paths.

### Explicit physical assumptions

Boundary conditions, gauge choice, parameter normalization and finite-size
limitations should remain visible in the implementation and documentation.

## Package structure

```text
QuantumSimulation/
├── algorithms/
│   ├── __init__.py
│   └── vqe.py
├── ansatz/
│   ├── __init__.py
│   ├── _builder.py
│   ├── hva.py
│   └── initial_state.py
├── circuits/
│   ├── __init__.py
│   ├── builder.py
│   └── transpilation.py
├── core/
│   ├── __init__.py
│   ├── loaders.py
│   └── types/
│       ├── __init__.py
│       ├── _generic.py
│       ├── algorithms.py
│       ├── ansatz.py
│       ├── circuits.py
│       ├── diagnostics.py
│       ├── dynamics.py
│       ├── hamiltonians.py
│       ├── hardware.py
│       └── mitigation.py
├── diagnostics/
│   ├── __init__.py
│   └── validator.py
├── dynamics/
│   ├── __init__.py
│   ├── builder.py
│   └── engine.py
├── hamiltonians/
│   ├── __init__.py
│   ├── _builder.py
│   └── schwinger.py
├── hardware/
│   ├── __init__.py
│   ├── _builder.py
│   ├── aer.py
│   ├── ibm_runtime.py
│   ├── ideal.py
│   └── noise.py
├── mitigation/
│   ├── __init__.py
│   └── richardson.py
├── observables/
│   ├── __init__.py
│   ├── _builder.py
│   ├── base.py
│   ├── charge.py
│   ├── electric_field.py
│   ├── energy.py
│   ├── gauss_law_violation.py
│   ├── pair_creation.py
│   └── persistence.py
├── operators/
│   ├── __init__.py
│   ├── _builder.py
│   ├── base.py
│   ├── fermions.py
│   └── gauge.py
├── processing/
│   ├── __init__.py
│   └── dataframes.py
├── SchwingerSimulation.py
├── Utils.py
├── _config.py
└── __init__.py
```

Tests are located in the repository-level `tests/` directory rather than
inside the package.

## High-level interface

The standard entry point is:

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
```

Depending on the configured workflow, the simulation may expose:

```python
simulation.initial_state
simulation.final_state

simulation.vacuum_energy
simulation.vacuum_parameters
simulation.vqe_result

simulation.ground_state_diagnostics

simulation.evolution_result
simulation.evolution_data
simulation.evolution_error

simulation.all_trotter_evolution_data
simulation.all_trotter_evolution_error
```

The exact set of populated attributes depends on whether the state is
provided directly, prepared variationally and evolved in time.

## `SchwingerSimulation`

`SchwingerSimulation` is the high-level orchestration class.

Its responsibilities are limited to coordinating the simulation workflow:

1. parsing configuration;
2. creating quantum primitives;
3. constructing the preparation Hamiltonian;
4. loading configured symmetry operators;
5. applying physical penalties or validations;
6. constructing the variational ansatz;
7. transpiling circuits when required;
8. preparing or accepting an initial state;
9. validating the prepared state;
10. delegating temporal evolution;
11. exposing the resulting structured output.

The class should not implement physical operators, observable formulas,
mitigation mathematics or notebook-specific result analysis directly.

A typical execution is:

```python
simulation = SchwingerSimulation(
    config=config,
    initial_state=None,
)

simulation.run_simulation(
    initial_vqe_params=None,
)
```

An already prepared state may also be supplied:

```python
simulation = SchwingerSimulation(
    config=config,
    initial_state=prepared_circuit,
)

simulation.run_simulation()
```

When primitives are active, the supplied state should normally be a
`QuantumCircuit`. Exact execution paths may also operate on a
`Statevector`.

## Core types

The `core/types/` package defines typed contracts shared across modules.

### Physical configuration

Relevant physical types include:

```python
SchwingerParams
HamiltonianConfig
```

`SchwingerParams` contains lattice and physical parameters such as:

```python
L
a
m
e0
periodic
```

The current gauge-eliminated Schwinger implementation supports open
boundary conditions.

### Hardware configuration

Hardware types include:

```python
HardwareConfig
NoiseConfig
NoiseErrorConfig
PrimitiveBundle
```

`PrimitiveBundle` groups backend and primitive resources:

```python
bundle.backend_type
bundle.backend
bundle.estimator
bundle.sampler
bundle.precision
bundle.shots
```

### Ansatz and optimization

Relevant types include:

```python
InitialStateConfig
AnsatzConfig
OptimizerConfig
VQEResult
```

`VQEResult` stores the main variational output:

```python
result.state
result.energy
result.parameters
result.optimization_result
result.optimization_history
result.duration
```

### Diagnostics

Physical validation is configured through:

```python
SymmetryCheckConfig
```

A check may define:

- operator or observable name;
- expected target value;
- numerical tolerance;
- penalty strength;
- whether the check is used during Hamiltonian preparation or state
  validation.

### Dynamics

The temporal-evolution types include:

```python
EvolutionConfig
EvolutionVariantConfig
QuenchConfig

ObservableSeries
TrajectoryResult
EvolutionResult
ExecutionMetrics
```

`EvolutionConfig` describes:

- total time;
- number of output intervals;
- evolution gate;
- product-formula synthesis;
- raw evolution variants;
- quench settings;
- observable selection;
- preferred trajectory;
- mitigation settings.

A typical configuration contains multiple variants:

```python
config.variants
```

Examples include:

```text
dt
dt_half
exact
```

Each `TrajectoryResult` stores:

```python
trajectory.key
trajectory.times
trajectory.observables
trajectory.final_state
trajectory.step_size
trajectory.metadata
```

`EvolutionResult` groups all raw and mitigated trajectories and identifies
the preferred output.

### Mitigation

Mitigation types include:

```python
MitigationConfig
MitigationStepConfig
MitigationRecord
```

Mitigation is configured as an ordered list of operations. The currently
implemented strategy is Richardson extrapolation.

## Hamiltonians

The `hamiltonians/` package contains complete physical Hamiltonian models.

The current implementation provides the Schwinger Hamiltonian in temporal
gauge:

```python
from QuantumSimulation.hamiltonians import (
    SchwingerHamiltonianTemporalGauge,
)
```

It can be created from typed parameters:

```python
from QuantumSimulation.core.types import (
    SchwingerParams,
)


params = SchwingerParams(
    L=4,
    a=0.5,
    m=0.2,
    e0=0.0,
    periodic=False,
)

hamiltonian = (
    SchwingerHamiltonianTemporalGauge(
        params
    )
)
```

The Hamiltonian contains physically labelled groups such as:

```text
hopping
mass
electric
```

The builder interface allows model selection from a
`HamiltonianConfig`:

```python
from QuantumSimulation.hamiltonians import (
    build_hamiltonian,
)


hamiltonian = build_hamiltonian(
    hamiltonian_config
)
```

Quenched Hamiltonians can be built using parameter overrides without
mutating the preparation configuration.

## Operators

The `operators/` package defines symbolic physical operators independently
from the observable measurement logic.

Current operators include:

- total charge;
- minimal or local charge components;
- electron and positron number;
- electric field;
- Gauss-law consistency;
- Schwinger Hamiltonian components.

All operators inherit from `BaseOperator`.

A `BaseOperator` stores physically labelled collections of sparse Pauli
terms:

```python
operator.terms_dict
operator.num_qubits
operator.params
```

Conversion to Qiskit is performed through:

```python
components = operator.to_qiskit(
    sparse=False
)

summed_operator = operator.to_qiskit(
    sparse=True
)
```

With `sparse=False`, the method preserves the physical components. With
`sparse=True`, all components are summed into one `SparsePauliOp`.

Operators can be built through the registry:

```python
from QuantumSimulation.operators import (
    build_operator,
)


charge = build_operator(
    "Total_Charge_Operator",
    params,
)
```

### Logical and physical layouts

Physical operators represent the logical lattice model.

When a circuit is transpiled to a concrete backend, Qiskit layout
transformations should be applied to the Qiskit operator representation
used for execution. Logical physical definitions should remain independent
from a particular hardware mapping.

## Observables

The `observables/` package defines how physical quantities are evaluated.

Current observables include:

```text
Energy
Total_Charge
Persistence
Pair_Creation
Electric_Field
Gauss_Law_Violation
```

Observables are created through a registry:

```python
from QuantumSimulation.observables import (
    build_observable,
)


observable = build_observable(
    name="Total_Charge",
    operator=charge_operator,
    precision=None,
)
```

### Observable interface

Observable implementations inherit from `BaseObservable`.

The relevant conceptual interface is:

```python
class BaseObservable:
    name: str
    primitive_type: str

    def get_operators(self):
        ...

    def get_pub(self, state):
        ...

    def calculate_exact(self, state):
        ...

    def process_result_data(
        self,
        values,
        uncertainties,
    ):
        ...

    def process_pub_result(self, result):
        ...
```

The supported primitive types are:

```text
estimator
sampler
```

### Exact evaluation

For statevectors or ideal circuits:

```python
value, uncertainty = (
    observable.calculate_exact(state)
)
```

Exact-state uncertainties are generally returned as zero because no
finite-shot sampling is involved.

### Primitive evaluation

Estimator observables build expectation-value PUBs:

```python
pub = observable.get_pub(circuit)
result = estimator.run([pub]).result()
value, uncertainty = (
    observable.process_pub_result(
        result[0]
    )
)
```

Sampler observables build measurement circuits and process quasi-count or
bit-array results.

### Scalar and multi-component observables

Observable results may be scalar or contain several physical components.

Examples:

```text
Energy
    scalar

Total_Charge
    scalar

Persistence
    scalar

Pair_Creation
    electrons, positrons

Electric_Field
    one value per physical link
```

Component labels are used by the processing layer to create stable column
names.

### Particle occupation and pair production

The particle-number operators return electron and positron occupations.

Pair production relative to the initial vacuum is a derived quantity:

\[
\Delta N_e(t)=N_e(t)-N_e(0),
\]

\[
\Delta N_p(t)=N_p(t)-N_p(0).
\]

The raw occupation of the staggered reference state is therefore not
necessarily zero. Creation relative to the initial state should be
calculated during physical post-processing.

### Gauss-law consistency

In the gauge-eliminated formulation, the electric field is reconstructed
from the charge distribution. Consequently, the local Gauss-law relation
is satisfied algebraically.

The current Gauss-law observable is therefore primarily a consistency
check of the reduced representation rather than a discriminator between
physical and unphysical states with independent gauge degrees of freedom.

## Ansatz construction

The `ansatz/` package provides initial-state and parameterized-circuit
construction.

### Initial states

The standard reference vacuum is generated through staggered occupation:

```python
from QuantumSimulation.ansatz import (
    build_initial_state,
)


initial_state = build_initial_state(
    num_qubits=params.L,
    config=initial_state_config,
)
```

### Standard ansatz families

The builder supports standard Qiskit circuit families configured through
`AnsatzConfig`, including hardware-efficient and excitation-preserving
forms.

### Hamiltonian variational ansatzes

The package also includes Hamiltonian variational ansatz variants that use
the physical components of the Schwinger Hamiltonian.

```python
from QuantumSimulation.ansatz import (
    build_parametrized_ansatz,
)


ansatz = build_parametrized_ansatz(
    num_qubits=params.L,
    config=ansatz_config,
    hamiltonian=hamiltonian,
)
```

Physics-informed ansatzes require the Hamiltonian to be supplied.

Ansatz builders produce logical circuits. Backend-specific compilation is
handled separately by the transpilation layer.

## VQE

The `algorithms/` package currently contains `VQESolver`.

```python
from QuantumSimulation.algorithms import (
    VQESolver,
)
```

The solver coordinates:

- initial-parameter generation;
- exact or estimator-based energy evaluation;
- optional estimator gradients;
- SciPy minimization;
- optimization-history collection;
- optional IBM Runtime sessions;
- final state binding;
- result construction.

The solver accepts explicit initial parameters:

```python
result = solver.solve(
    initial_params=parameters,
)
```

Otherwise, initial parameters are generated according to the configured
strategy.

Possible strategies include:

```text
zeros
random_small
uniform_random
```

For reproducible variational studies, random seeds and initial parameters
should be recorded.

## Diagnostics and validation

`PhysicsValidator` performs Hamiltonian and state-level physical checks.

### Hamiltonian checks

Examples include:

- commutation with total charge;
- symmetry compatibility;
- penalty construction;
- coefficient-scale inspection.

### State validation

Configured observables can be evaluated against expected values and
tolerances.

```python
report = validator.validate_state(
    state=prepared_state,
    checks=validation_checks,
)
```

A validation report may contain:

```python
{
    "Total_Charge": {
        "value": 0.0,
        "error": 0.0,
        "target": 0.0,
        "tolerance": 1e-8,
        "passed": True,
    }
}
```

Symmetry checks, gauge identities and expected state properties should be
interpreted according to their distinct physical meanings.

## Hardware and primitives

The `hardware/` package constructs execution resources from
`HardwareConfig`.

```python
from QuantumSimulation.hardware import (
    setup_quantum_primitives,
)


primitives = setup_quantum_primitives(
    hardware_config
)
```

### Direct execution

When no estimator or sampler is configured, compatible observables are
evaluated from statevectors.

This is appropriate for exact validation and small systems.

### Ideal primitives

Qiskit statevector primitives provide estimator and sampler interfaces
without hardware noise.

They are useful for validating PUB construction and result-processing
paths.

### Aer

Aer execution supports:

- finite-shot simulation;
- backend-oriented transpilation;
- configurable noise models;
- estimator and sampler primitives.

Noise configuration is handled independently from the physical model.

### IBM Runtime

IBM Runtime support is available as an optional execution path.

Remote execution requires:

- installation of the IBM Runtime dependency;
- valid credentials;
- access to a compatible backend;
- explicit selection of the IBM Runtime backend type.

IBM Runtime tests are marked separately and excluded from the default test
and continuous-integration commands.

## Circuits and transpilation

The `circuits/` package contains configurable circuit construction and
backend compilation.

### Circuit builder

The circuit builder supports reusable configured operations, including:

- single-qubit gates;
- controlled gates;
- parameterized rotations;
- logical qubit references;
- circuit composition.

### Transpilation

Backend-aware transpilation is applied after constructing logical circuits.

The transpilation layer coordinates:

- backend selection;
- optimization level;
- initial layout;
- compatibility between preparation and evolution circuits;
- operator layout for primitive execution.

Logical physical objects should remain distinguishable from their
backend-specific ISA representations.

## Temporal evolution

Temporal evolution is implemented in the `dynamics/` package.

The dynamics layer receives:

- an initial state;
- an evolution Hamiltonian;
- typed evolution configuration;
- configured observables;
- primitive resources.

It returns an `EvolutionResult`.

### Quenches

A quench changes selected Hamiltonian parameters before temporal
evolution.

```python
quench.parameter_overrides
```

The preparation Hamiltonian remains unchanged. A new evolution Hamiltonian
is constructed from the overridden parameters.

### Evolution variants

Each configured evolution variant defines a numerical trajectory.

Examples:

```text
dt
dt_half
exact
```

A Trotter variant contains a step multiplier relative to the base output
interval:

\[
\Delta t_{\text{variant}}
=
\Delta t_{\text{base}}
\times
\text{step multiplier}.
\]

The internal step multiplier must divide the base interval exactly, so
each variant produces output on the same external time grid.

### Exact matrix evolution

For small systems, the evolution Hamiltonian can be applied directly using
matrix-exponential methods.

Exact matrix evolution provides a reference for:

- Trotter convergence;
- norm conservation;
- observable trajectories;
- persistence;
- final-state fidelity.

### Trotterized evolution

Circuit-based evolution uses Pauli evolution gates and configurable
Suzuki-Trotter synthesis.

Multiple step sizes can be evaluated in one run to characterize
discretization effects.

### Measurement execution

The dynamics engine supports:

- exact observable evaluation;
- estimator PUB grouping;
- sampler PUB execution;
- result decoding;
- timing and workload metadata.

Estimator and sampler operations are tracked independently.

### State storage

The final physical state is retained for each raw trajectory.

Intermediate states should only be stored when explicitly requested,
because statevectors scale exponentially and cumulative circuits grow in
depth over time.

## Mitigation

Algorithmic mitigation is implemented independently from temporal
evolution.

The current method is Richardson extrapolation.

Assuming:

\[
Q(h)=Q_0+c h^p+\mathcal{O}(h^{p+1}),
\]

and two compatible trajectories with step sizes \(h_c>h_f\), the corrected
observable estimate is:

\[
Q_R=
\frac{r^pQ_f-Q_c}{r^p-1},
\qquad
r=\frac{h_c}{h_f}.
\]

Values and uncertainties are processed separately.

For statistically independent estimates, the propagated uncertainty is:

\[
\sigma_R=
\sqrt{
\left(
\frac{r^p}{r^p-1}\sigma_f
\right)^2
+
\left(
\frac{1}{r^p-1}\sigma_c
\right)^2
}.
\]

A configurable correlation term may be introduced when coarse and fine
estimates are statistically correlated.

Richardson extrapolation generates a corrected observable trajectory. It
does not generate a corrected statevector or quantum circuit. The final
state associated with a Richardson result is therefore taken from the
finest underlying raw trajectory.

## Processing

The `processing/` package converts structured trajectories into
analysis-friendly formats.

The principal processing output consists of two aligned DataFrames:

```python
values_dataframe
uncertainties_dataframe
```

Both are indexed by physical time and contain the same column order.

Example columns include:

```text
Energy
Total_Charge
Persistence
Pair_Creation_electrons
Pair_Creation_positrons
Electric_Field_link_0
Electric_Field_link_1
Gauss_Law_Violation
```

The processing layer expands multi-component observables using the
component labels defined by their observable implementations.

Notebook-specific derived quantities, fits and plots should remain outside
the simulation package unless they represent generally reusable physical
operations.

## Configuration

Simulations are normally configured through JSON files located in the
repository-level `configs/` directory.

A simplified example is:

```json
{
  "QubitsNumber": 4,

  "Hamiltonian": {
    "Type": "Schwinger",
    "Gauge": "Temporal",
    "Parameters": {
      "L": 4,
      "a": 0.5,
      "m": 0.2,
      "e0": 0.0,
      "periodic": false
    }
  },

  "Backend": {
    "Type": "StatevectorEstimator",
    "Shots": 4096,
    "Precision": null
  },

  "Ansatz": {
    "Type": "ExcitationPreserving",
    "Reps": 1,

    "Initial State": {
      "Vacuum": true,
      "Staggered": true
    },

    "Init_Strategy": "zeros",
    "Use Gradient Cost": false,

    "Minimizer": {
      "Method": "COBYLA",
      "Options": {
        "maxiter": 200
      }
    }
  },

  "Diagnostics": {
    "Symmetries": [],
    "StateValidation": []
  },

  "Temporal Evolution": {
    "Active": true,
    "Total_Time": 2.0,
    "Time_Steps": 20,

    "Evolution_Gate": {
      "Type": "Pauli",
      "Synthesis": "TrotterSuzuki",
      "Synthesis_Params": {
        "order": 2,
        "reps": 1
      }
    },

    "Variants": {
      "dt": {
        "Method": "Trotter",
        "Step_Multiplier": 1.0
      },

      "dt_half": {
        "Method": "Trotter",
        "Step_Multiplier": 0.5
      }
    },

    "Preferred_Variant": "richardson",

    "Quench": {
      "Active": true,
      "Parameters_to_Change": {
        "e0": 0.5
      }
    },

    "Observables": {
      "Observables_List": [
        "Energy",
        "Persistence",
        "Pair_Creation",
        "Electric_Field"
      ],

      "Observables_Params": {}
    },

    "Mitigation": {
      "Enabled": true,
      "Steps": [
        {
          "Method": "Richardson",
          "Options": {
            "coarse_key": "dt",
            "fine_key": "dt_half",
            "order": 2,
            "output_key": "richardson"
          }
        }
      ],

      "Preferred_Output": "richardson"
    },

    "Store_States": false
  }
}
```

Use the current repository configurations as the executable reference for
the precise accepted field names.

## Result access

After a ground-state-only execution:

```python
simulation.run_simulation()

print(simulation.vacuum_energy)
print(simulation.ground_state_diagnostics)
```

After temporal evolution:

```python
print(simulation.evolution_data.head())
print(simulation.evolution_error.head())
```

All raw variants remain available:

```python
simulation.all_trotter_evolution_data[
    "dt"
]

simulation.all_trotter_evolution_data[
    "dt_half"
]
```

When mitigation is active:

```python
simulation.all_trotter_evolution_data[
    "richardson"
]
```

The structured result provides the preferred trajectory and final physical
state:

```python
result = simulation.evolution_result

preferred = result.preferred
final_state = result.final_state
```

## Testing

Tests are located at repository level:

```text
tests/
├── conftest.py
├── unit/
└── integration/
```

Run the default suite from the repository root:

```bash
python -m pytest \
    -m "not slow and not ibm"
```

Run with coverage:

```bash
python -m pytest \
    -m "not slow and not ibm" \
    --cov=QuantumSimulation \
    --cov-report=term-missing \
    --cov-report=html
```

Tests currently cover:

- typed configuration and result contracts;
- Hamiltonian construction;
- physical operators;
- hermiticity and conserved charge;
- exact observables;
- estimator and sampler paths;
- ansatz construction;
- VQE;
- hardware builders;
- Aer noise;
- physical diagnostics;
- dynamics configuration;
- Richardson extrapolation;
- high-level simulation orchestration.

IBM Runtime tests use the `ibm` marker and are intentionally excluded from
the standard command and the default continuous-integration workflow.

## Extending the package

### Adding a Hamiltonian

1. Implement a physical model under `hamiltonians/`.
2. Define a typed parameter object if the model requires new parameters.
3. Return a `BaseOperator`-compatible representation.
4. Register the model in the Hamiltonian builder.
5. Add hermiticity, dimension and symmetry tests.
6. Document boundary conditions and normalization.

### Adding an operator

1. Create a `BaseOperator` subclass.
2. Define labelled physical components.
3. Register the operator.
4. Add exact small-system matrix tests.
5. Verify the expected number of components and qubits.

### Adding an observable

1. Create a `BaseObservable` subclass.
2. Select `estimator` or `sampler`.
3. Define the required operator or reference state.
4. Implement exact evaluation when possible.
5. Implement primitive-result processing.
6. Define component labels for vector observables.
7. Register the observable and its operator dependency.
8. Add exact and primitive-based tests.

### Adding an ansatz

1. Implement a logical circuit builder under `ansatz/`.
2. Do not perform backend-specific transpilation in the builder.
3. Register the ansatz type.
4. Validate qubit count and parameter count.
5. Add a small VQE or state-preparation test.

### Adding a dynamics method

1. Add a normalized method name to the evolution builder.
2. Implement trajectory generation in `dynamics/`.
3. Preserve the common external time grid.
4. Return structured trajectory results.
5. Track the final state and execution metadata.
6. Validate against exact small-system evolution.

### Adding mitigation

1. Implement the correction under `mitigation/`.
2. Operate on compatible trajectory results.
3. preserve raw trajectories;
4. create a new derived trajectory;
5. propagate uncertainties independently from values;
6. record source trajectories and method options;
7. define the physical source of the final state;
8. add synthetic numerical tests.

## Development guidelines

When modifying the package:

- keep raw JSON parsing at the orchestration boundary;
- use typed configuration internally;
- keep physical operators independent from hardware layouts;
- avoid notebook-specific calculations inside the simulation engine;
- preserve raw and mitigated results separately;
- distinguish occupation from creation relative to a reference state;
- distinguish statistical uncertainty from systematic algorithmic error;
- avoid silently mutating physical configuration;
- keep exact small-system references;
- add tests for every new physical convention;
- use clear errors for unsupported execution combinations;
- document gauge, boundaries and normalization explicitly.

## Physical and numerical limitations

The package currently has several important limitations:

- the primary physical model is the massive Schwinger model;
- the gauge-eliminated formulation uses open boundary conditions;
- exact matrix and statevector methods scale exponentially;
- variational convergence depends on initialization, ansatz and optimizer;
- circuit depth grows with evolution time;
- finite-shot uncertainty does not include every systematic error;
- Richardson extrapolation assumes a valid asymptotic error regime;
- mitigation of Trotter error is distinct from hardware-noise mitigation;
- gauge-law checks are algebraic in the reduced formulation;
- calculations at finite lattice size and spacing do not directly establish
  continuum or thermodynamic results;
- remote hardware execution requires additional configuration and
  validation.

No claim of quantum advantage is made.

## Reproducibility recommendations

For every numerical study, record:

```text
repository commit or release
Python version
dependency versions
configuration file
random seeds
physical parameters
boundary conditions
backend
number of shots
precision
ansatz
optimizer
initial variational parameters
quench definition
time grid
Trotter synthesis
Trotter step sizes
mitigation configuration
```

When results are loaded from previous runs, record the original execution
metadata and package version.

## Related documentation

For project-level installation, research workflows, citation information
and the overall roadmap, see:

```text
../README.md
```

For executable parameter examples, see:

```text
../configs/
```

For research notebooks and study-specific analysis, see:

```text
../results/
```

For tests, see:

```text
../tests/
```
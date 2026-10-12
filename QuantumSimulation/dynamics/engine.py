# QuantumSimulation/dynamics/engine.py
import time
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm.auto import tqdm
from scipy.sparse.linalg import expm_multiply
from qiskit.quantum_info import Statevector

from QuantumSimulation.core.types import (
    EvolutionConfig,
    EvolutionVariantConfig,
    EvolutionResult,
    TrajectoryResult,
    ObservableSeries,
    ExecutionMetrics,
    QuantumState
)
from QuantumSimulation.core.types import PrimitiveBundle
from QuantumSimulation.Utils import getTimer

# =====================================================================
# Internal Type Definitions for State & Data Tracking
# =====================================================================

@dataclass
class ObservableAccumulator:
    """Internal accumulator for sequential observable evaluations."""
    name: str
    values: list[Any] = field(default_factory=list)
    uncertainties: list[Any] = field(default_factory=list)

@dataclass
class TrajectoryTracker:
    """Internal tracker for a single trajectory variant during evolution."""
    variant: EvolutionVariantConfig
    current_state: QuantumState
    accumulators: dict[str, ObservableAccumulator] = field(default_factory=dict)

@dataclass
class _PubBatch:
    """Internal helper to organize Qiskit Primitives PUBs."""
    pubs: list[Any] = field(default_factory=list)
    observables: list[list[Any]] = field(default_factory=list)
    lengths: list[list[int]] = field(default_factory=list)
    variant_keys: list[str] = field(default_factory=list)


# =====================================================================
# Main Engine Orchestrator
# =====================================================================

def run_time_evolution(
    initial_state: QuantumState,
    hamiltonian: Any,
    observables_objects: list[Any],
    config: EvolutionConfig,
    primitives: PrimitiveBundle,
    num_qubits: int,
    evolution_circuits: dict[str, Any]
) -> EvolutionResult:
    """
    Executes the temporal evolution loop, dispatches physics updates,
    and returns a strictly typed EvolutionResult domain model.

    Parameters
    ----------
    initial_state: QuantumState
        The starting quantum state for all trajectory variants.
    hamiltonian: Any
        The Hamiltonian operator used for evolution.
    observables_objects: list[Any]
        List of observable objects to evaluate at each time step.
    config: EvolutionConfig
        Configuration for the evolution, including time steps and variants.
    primitives: PrimitiveBundle
        Hardware or simulator primitives for executing quantum circuits.
    num_qubits: int
        Number of qubits in the system.
    evolution_circuits: dict[str, Any]
        Pre-built evolution circuits for each trajectory variant.
    
    Returns
    -------
    EvolutionResult
        Strictly typed result containing trajectories, observables, and metrics.
    """
    metrics = ExecutionMetrics()
    
    # 1. Initialize Typed Trackers
    trackers = _initialize_trackers(config, initial_state, observables_objects)

    # Pre-compute sparse matrix if exact evolution is present in any variant
    sparse_ham = None
    if any(v.method == "matrix_exponential" for v in config.variants):
        sparse_ham = hamiltonian.to_matrix(sparse=True)

    estimator_batch = _PubBatch()
    sampler_batch = _PubBatch()

    # 2. Main Time Loop
    evol_start_time = time.time()
    
    with tqdm(range(config.time_steps + 1), desc="Evolving state", unit="step", file=sys.stdout, dynamic_ncols=False) as pbar:
        for t in pbar:
            for variant in config.variants:
                tracker = trackers[variant.key]
                
                # A. Collect Observables (Evaluated directly or deferred to PUBs)
                _collect_observables(
                    tracker, observables_objects, primitives, estimator_batch, sampler_batch
                )

                # B. Advance Quantum State (Skip on the final measurement step)
                if t < config.time_steps:
                    tracker.current_state = _advance_state(
                        state=tracker.current_state,
                        variant=variant,
                        base_step=config.base_step,
                        num_qubits=num_qubits,
                        sparse_ham=sparse_ham,
                        circuit=evolution_circuits.get(variant.key),
                        is_ideal=(primitives.estimator is None)
                    )

    metrics.evolution_duration = time.time() - evol_start_time

    # 3. Execute Hardware / Qiskit Primitives Batches
    _execute_estimator_batch(primitives, estimator_batch, trackers, metrics)
    _execute_sampler_batch(primitives, sampler_batch, trackers, metrics)

    # 4. Assemble Final Domain Models
    return _build_evolution_result(trackers, config, metrics)


# =====================================================================
# Private Engine Components & Delegates
# =====================================================================

def _initialize_trackers(
    config: EvolutionConfig, 
    initial_state: QuantumState, 
    observables_objects: list[Any]
) -> dict[str, TrajectoryTracker]:
    """Sets up state and data accumulators for all requested variants."""
    trackers = {}
    for variant in config.variants:
        accs = {obs.name: ObservableAccumulator(obs.name) for obs in observables_objects}
        trackers[variant.key] = TrajectoryTracker(
            variant=variant,
            current_state=initial_state.copy(),
            accumulators=accs
        )
    return trackers


def _advance_state(
    state: QuantumState,
    variant: EvolutionVariantConfig,
    base_step: float,
    num_qubits: int,
    sparse_ham: Any,
    circuit: Any,
    is_ideal: bool
) -> QuantumState:
    """
    Advances a quantum state purely based on the variant's configured method.
    Decouples matrix multiplication from circuit execution cleanly.
    """
    applications = round(1.0 / variant.step_multiplier)
    
    if variant.method == "matrix_exponential":
        dt = base_step * variant.step_multiplier
        state_data = state.data
        for _ in range(applications):
            state_data = expm_multiply(-1j * sparse_ham * dt, state_data)
        return Statevector(state_data)
        
    elif variant.method == "trotter":
        for _ in range(applications):
            if is_ideal:
                state = state.evolve(circuit)
            else:
                state.compose(circuit, range(num_qubits), inplace=True)
        return state
        
    else:
        raise ValueError(f"Evolution method '{variant.method}' is not supported.")


def _collect_observables(
    tracker: TrajectoryTracker, 
    observables_objects: list[Any],
    primitives: PrimitiveBundle, 
    estimator_batch: _PubBatch, 
    sampler_batch: _PubBatch
) -> None:
    """Evaluates ideal observables immediately or groups them into PUB batches."""
    step_operators, step_obs_objects, step_lengths = [], [], []

    for obs in observables_objects:
        if primitives.estimator is None:
            # Ideal exact execution
            val, err = obs.calculate_exact(tracker.current_state)
            tracker.accumulators[obs.name].values.append(val)
            tracker.accumulators[obs.name].uncertainties.append(err)
        else:
            # Hardware PUB generation
            pub = obs.get_pub(tracker.current_state)
            if obs.primitive_type == "estimator":
                operators = obs.get_operators()
                step_operators.extend(operators)
                step_obs_objects.append(obs)
                step_lengths.append(len(operators))
            elif obs.primitive_type == "sampler":
                sampler_batch.pubs.append(pub)
                sampler_batch.observables.append(obs)
                sampler_batch.variant_keys.append(tracker.variant.key)

    # Flush step operators to the Estimator batch
    if step_operators:
        grouped_pub = (tracker.current_state, step_operators, None, primitives.precision)
        estimator_batch.pubs.append(grouped_pub)
        estimator_batch.observables.append(step_obs_objects)
        estimator_batch.lengths.append(step_lengths)
        estimator_batch.variant_keys.append(tracker.variant.key)


def _execute_estimator_batch(
    primitives: PrimitiveBundle, 
    batch: _PubBatch, 
    trackers: dict[str, TrajectoryTracker], 
    metrics: ExecutionMetrics
) -> None:
    """Executes deferred Estimator PUBs and maps results back to trackers."""
    if primitives.estimator is None or not batch.pubs:
        return

    print(f"{getTimer()} INFO: Dispatching {len(batch.pubs)} grouped Estimator PUBs...")
    start_time = time.time()
    results = primitives.estimator.run(batch.pubs).result()
    
    metrics.estimator_duration = time.time() - start_time
    metrics.estimator_pub_count = len(batch.pubs)

    for grouped_result, obs_list, lengths, variant_key in zip(
        results, batch.observables, batch.lengths, batch.variant_keys
    ):
        evs_all = np.atleast_1d(grouped_result.data.evs)
        stds_all = np.atleast_1d(grouped_result.data.stds)

        current_idx = 0
        for obs_obj, length in zip(obs_list, lengths):
            evs_slice = evs_all[current_idx : current_idx + length]
            stds_slice = stds_all[current_idx : current_idx + length]
            
            val, err = obs_obj.process_result_data(evs_slice, stds_slice)
            trackers[variant_key].accumulators[obs_obj.name].values.append(val)
            trackers[variant_key].accumulators[obs_obj.name].uncertainties.append(err)
            
            current_idx += length


def _execute_sampler_batch(
    primitives: PrimitiveBundle, 
    batch: _PubBatch, 
    trackers: dict[str, TrajectoryTracker], 
    metrics: ExecutionMetrics
) -> None:
    """Executes deferred Sampler PUBs and maps results back to trackers."""
    if primitives.sampler is None or not batch.pubs:
        return

    print(f"{getTimer()} INFO: Dispatching {len(batch.pubs)} Sampler PUBs...")
    start_time = time.time()
    results = primitives.sampler.run(batch.pubs).result()
    
    metrics.sampler_duration = time.time() - start_time
    metrics.sampler_pub_count = len(batch.pubs)

    for result, obs_obj, variant_key in zip(results, batch.observables, batch.variant_keys):
        val, err = obs_obj.process_pub_result(result)
        trackers[variant_key].accumulators[obs_obj.name].values.append(val)
        trackers[variant_key].accumulators[obs_obj.name].uncertainties.append(err)


def _build_evolution_result(
    trackers: dict[str, TrajectoryTracker], 
    config: EvolutionConfig, 
    metrics: ExecutionMetrics
) -> EvolutionResult:
    """
    Parses the accumulated native Python lists into strict Domain Models 
    (ObservableSeries, TrajectoryResult, EvolutionResult).
    """
    trajectories = {}
    time_grid = config.times

    for key, tracker in trackers.items():
        observables_dict = {}
        
        for obs_name, acc in tracker.accumulators.items():
            # Apply user-defined options/labels if they exist for this observable
            obs_opts = config.observable_options.get(obs_name, {})
            labels = obs_opts.get("component_labels", ())
            
            series = ObservableSeries(
                values=np.array(acc.values),
                uncertainties=np.array(acc.uncertainties),
                component_labels=tuple(labels)
            )
            observables_dict[obs_name] = series

        trajectories[key] = TrajectoryResult(
            key=key,
            times=time_grid,
            observables=observables_dict,
            final_state=tracker.current_state,
            step_size=config.base_step * tracker.variant.step_multiplier
        )

    pref_key = config.preferred_variant or config.variants[0].key

    return EvolutionResult(
        trajectories=trajectories,
        preferred_key=pref_key,
        metrics=metrics
    )
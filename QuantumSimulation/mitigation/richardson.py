# QuantumSimulation/postprocessing/mitigation.py

from __future__ import annotations

import numpy as np

from QuantumSimulation.core.types import (
    EvolutionResult, 
    TrajectoryResult, 
    ObservableSeries,
    MitigationRecord
)

from QuantumSimulation.core.types import MitigationConfig

def apply_mitigation_pipeline(
    result: EvolutionResult, 
    config: MitigationConfig
) -> EvolutionResult:
    """
    Applies the sequence of mitigation strategies defined in the configuration.
    Mutates the EvolutionResult by adding new mitigated TrajectoryResults.

    Parameters
    ----------
    result: EvolutionResult
        The original evolution result containing trajectories to be mitigated.
    config: MitigationConfig
        The mitigation configuration containing the steps to apply.

    Returns
    -------
    EvolutionResult
        The evolution result with mitigated trajectories added.
    """
    if not config.enabled or not config.steps:
        return result

    for step in config.steps:
        method = step.method.lower()
        if method == "richardson":
            _apply_richardson(result, step.options)
        elif method == "zne":
            # Future extension placeholder
            raise NotImplementedError("ZNE mitigation not yet implemented.")
        else:
            raise ValueError(f"Unknown mitigation method: {step.method}")

    # Update preferred key if specified in config
    if config.preferred_output and config.preferred_output in result.trajectories:
        result.preferred_key = config.preferred_output

    return result

def _apply_richardson(result: EvolutionResult, options: dict) -> None:
    """
    Richardson Extrapolation for two trajectory variants.
    Expects 'coarse_key', 'fine_key', 'order', and 'output_key' in options.
    """
    coarse_key = options["coarse_key"]
    fine_key = options["fine_key"]
    order = options.get("order", 2)
    output_key = options.get("output_key", "mitigated_richardson")

    coarse_traj = result.trajectories[coarse_key]
    fine_traj = result.trajectories[fine_key]

    # Calculate step ratio (r) and scaling factor (factor)
    r = coarse_traj.step_size / fine_traj.step_size
    factor = r ** order

    mitigated_observables = {}
    for obs_name in fine_traj.observables:
        fine_series = fine_traj.observables[obs_name]
        coarse_series = coarse_traj.observables[obs_name]

        # Mitigate values: (r^p * O_fine - O_coarse) / (r^p - 1)
        mit_values = (factor * fine_series.values - coarse_series.values) / (factor - 1)

        # Propagate uncertainties in quadrature
        term_fine = (factor / (factor - 1)) ** 2 * (fine_series.uncertainties ** 2)
        term_coarse = (1 / (factor - 1)) ** 2 * (coarse_series.uncertainties ** 2)
        mit_uncertainties = np.sqrt(term_fine + term_coarse)

        mitigated_observables[obs_name] = ObservableSeries(
            values=mit_values,
            uncertainties=mit_uncertainties,
            component_labels=fine_series.component_labels,
            metadata={"mitigated": True, "method": "richardson"}
        )

    # Create new trajectory pointing to the fine trajectory's state
    result.trajectories[output_key] = TrajectoryResult(
        key=output_key,
        times=fine_traj.times,
        observables=mitigated_observables,
        final_state=None,
        state_source_key=fine_key,
        step_size=None,
        metadata={"mitigation": "richardson"}
    )

    # Record provenance
    result.mitigation_history.append(
        MitigationRecord(
            method="richardson",
            source_keys=(coarse_key, fine_key),
            output_key=output_key,
            options=options
        )
    )
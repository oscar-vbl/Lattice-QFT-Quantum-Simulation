# QuantumSimulation/core/types/dynamics.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .mitigation import MitigationConfig


QuantumState: TypeAlias = QuantumCircuit | Statevector


@dataclass(frozen=True)
class QuenchConfig:
    """
    Configuration for an instantaneous Hamiltonian quench.

    Parameters
    ----------
    active:
        Whether the post-preparation Hamiltonian differs from the
        preparation Hamiltonian.
    parameter_overrides:
        Hamiltonian parameters changed at the beginning of the
        temporal evolution.
    """

    active: bool = False
    parameter_overrides: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.active and self.parameter_overrides:
            raise ValueError(
                "Quench parameter overrides were provided while "
                "the quench is disabled."
            )


@dataclass(frozen=True)
class EvolutionVariantConfig:
    """
    Configuration for one evolution trajectory.

    A variant identifies a concrete numerical strategy. For
    Trotter evolution, ``step_multiplier`` scales the base time
    interval.

    Examples
    --------
    ``step_multiplier=1.0`` represents ``dt``.
    ``step_multiplier=0.5`` represents ``dt / 2``.
    """

    key: str
    method: str = "trotter"
    step_multiplier: float = 1.0
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_key = self.key.strip()
        normalized_method = self.method.strip().lower()

        if not normalized_key:
            raise ValueError(
                "Evolution variant key cannot be empty."
            )

        if not normalized_method:
            raise ValueError(
                "Evolution method cannot be empty."
            )

        if self.step_multiplier <= 0:
            raise ValueError(
                "step_multiplier must be greater than zero."
            )

        object.__setattr__(self, "key", normalized_key)
        object.__setattr__(
            self,
            "method",
            normalized_method,
        )


@dataclass(frozen=True)
class EvolutionConfig:
    """
    Typed configuration for temporal evolution.

    Parameters
    ----------
    active:
        Whether temporal evolution is executed.
    total_time:
        Final physical evolution time.
    time_steps:
        Number of base intervals between zero and total_time.
    gate_type:
        Evolution gate implementation.
    synthesis:
        Product-formula synthesis strategy.
    synthesis_options:
        Options forwarded to the synthesis builder.
    variants:
        Evolution trajectories that will be generated.
    preferred_variant:
        Preferred raw or mitigated result key.
    quench:
        Optional parameter quench.
    observables:
        Observable names evaluated along each trajectory.
    observable_options:
        Observable-specific configuration.
    mitigation:
        Optional mitigation pipeline configuration.
    store_states:
        Whether all intermediate states should be retained.
        Keeping this disabled avoids storing many statevectors or
        increasingly deep circuits.
    """

    active: bool
    total_time: float
    time_steps: int

    gate_type: str = "pauli"
    synthesis: str = "suzuki_trotter"
    synthesis_options: dict[str, Any] = field(
        default_factory=dict
    )

    variants: tuple[EvolutionVariantConfig, ...] = (
        EvolutionVariantConfig(key="dt"),
    )

    preferred_variant: str | None = None
    quench: QuenchConfig = field(
        default_factory=QuenchConfig
    )

    observables: tuple[str, ...] = ()
    observable_options: dict[str, dict[str, Any]] = (
        field(default_factory=dict)
    )

    mitigation: MitigationConfig | None = None
    store_states: bool = False

    def __post_init__(self) -> None:
        if self.total_time <= 0:
            raise ValueError(
                "total_time must be greater than zero."
            )

        if self.time_steps <= 0:
            raise ValueError(
                "time_steps must be greater than zero."
            )

        if not self.variants:
            raise ValueError(
                "At least one evolution variant is required."
            )

        keys = [
            variant.key
            for variant in self.variants
        ]

        if len(keys) != len(set(keys)):
            raise ValueError(
                "Evolution variant keys must be unique."
            )

        for variant in self.variants:
            if variant.method == "trotter":
                inverse_multiplier = (
                    1.0 / variant.step_multiplier
                )
                applications = round(
                    inverse_multiplier
                )

                if not np.isclose(
                    inverse_multiplier,
                    applications,
                ):
                    raise ValueError(
                        "Trotter step_multiplier must divide "
                        "the base interval exactly. "
                        f"Variant '{variant.key}' uses "
                        f"{variant.step_multiplier}."
                    )

        if (
            self.preferred_variant is not None
            and self.preferred_variant not in keys
        ):
            mitigation_outputs = (
                self._configured_mitigation_outputs()
            )

            if (
                self.preferred_variant
                not in mitigation_outputs
            ):
                raise ValueError(
                    "preferred_variant must reference either "
                    "a raw evolution variant or a configured "
                    "mitigation output."
                )

        unknown_observable_options = (
            set(self.observable_options)
            - set(self.observables)
        )

        if unknown_observable_options:
            raise ValueError(
                "Observable options were provided for inactive "
                "observables: "
                f"{sorted(unknown_observable_options)}"
            )

    @property
    def base_step(self) -> float:
        """
        Base physical time interval.
        """
        return self.total_time / self.time_steps

    @property
    def times(self) -> np.ndarray:
        """
        Shared output time grid.

        All variants must produce results on this grid, even if
        they use smaller internal integration steps.
        """
        return np.linspace(
            0.0,
            self.total_time,
            self.time_steps + 1,
        )

    def get_variant(
        self,
        key: str,
    ) -> EvolutionVariantConfig:
        """
        Return one configured evolution variant.
        """
        for variant in self.variants:
            if variant.key == key:
                return variant

        raise KeyError(
            f"Evolution variant '{key}' not found."
        )

    def _configured_mitigation_outputs(
        self,
    ) -> set[str]:
        if self.mitigation is None:
            return set()

        outputs = set()

        for step in self.mitigation.steps:
            output_key = step.options.get(
                "output_key"
            )

            if output_key is not None:
                outputs.add(output_key)

        return outputs


@dataclass(frozen=True)
class ObservableResult:
    """
    Value and uncertainty of one observable at one time.
    """

    value: Any
    uncertainty: Any


@dataclass
class ObservableSeries:
    """
    Time series for one observable.

    ``values`` and ``uncertainties`` must have the same shape.
    Their first dimension corresponds to time. Remaining
    dimensions represent observable components.
    """

    values: np.ndarray
    uncertainties: np.ndarray
    component_labels: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)
        self.uncertainties = np.asarray(
            self.uncertainties
        )

        if (
            self.values.shape
            != self.uncertainties.shape
        ):
            raise ValueError(
                "Observable values and uncertainties must "
                "have the same shape."
            )

        if self.values.ndim == 0:
            raise ValueError(
                "ObservableSeries requires a time dimension."
            )

        if self.values.ndim == 1:
            if self.component_labels:
                raise ValueError(
                    "Scalar observable series must not define "
                    "component labels."
                )
            return

        component_count = self.values.shape[1]

        if (
            self.component_labels
            and len(self.component_labels)
            != component_count
        ):
            raise ValueError(
                "The number of component labels does not "
                "match the observable component dimension."
            )


@dataclass
class TrajectoryResult:
    """
    Raw or derived observable trajectory.

    A Richardson trajectory may contain extrapolated observable
    values without representing a physical quantum state. In that
    case, ``final_state`` is None and ``state_source_key`` points
    to the raw trajectory whose final state should be used.
    """

    key: str
    times: np.ndarray
    observables: dict[str, ObservableSeries]

    final_state: QuantumState | None = None
    state_source_key: str | None = None
    step_size: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.times = np.asarray(
            self.times,
            dtype=float,
        )

        if self.times.ndim != 1:
            raise ValueError(
                "Trajectory times must be one-dimensional."
            )

        if len(self.times) == 0:
            raise ValueError(
                "Trajectory must contain at least one time."
            )

        if np.any(np.diff(self.times) <= 0):
            raise ValueError(
                "Trajectory times must be strictly increasing."
            )

        for name, observable in (
            self.observables.items()
        ):
            if observable.values.shape[0] != len(
                self.times
            ):
                raise ValueError(
                    f"Observable '{name}' contains "
                    f"{observable.values.shape[0]} samples, "
                    f"but the trajectory contains "
                    f"{len(self.times)} times."
                )

        if (
            self.final_state is None
            and self.state_source_key is None
        ):
            raise ValueError(
                "A trajectory without final_state must define "
                "state_source_key."
            )


@dataclass
class ExecutionMetrics:
    """
    Execution timing and workload information.
    """

    evolution_duration: float = 0.0
    estimator_duration: float = 0.0
    sampler_duration: float = 0.0

    estimator_pub_count: int = 0
    sampler_pub_count: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class MitigationRecord:
    """
    Provenance record for one mitigation operation.
    """

    method: str
    source_keys: tuple[str, ...]
    output_key: str
    options: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class EvolutionResult:
    """
    Complete output of the dynamics pipeline.
    """

    trajectories: dict[str, TrajectoryResult]
    preferred_key: str
    metrics: ExecutionMetrics = field(
        default_factory=ExecutionMetrics
    )
    mitigation_history: list[MitigationRecord] = (
        field(default_factory=list)
    )

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise ValueError(
                "EvolutionResult requires at least one "
                "trajectory."
            )

        if (
            self.preferred_key
            not in self.trajectories
        ):
            raise ValueError(
                f"Preferred trajectory "
                f"'{self.preferred_key}' does not exist."
            )

        self._validate_state_sources()

    @property
    def preferred(self) -> TrajectoryResult:
        return self.trajectories[
            self.preferred_key
        ]

    @property
    def final_state(self) -> QuantumState:
        trajectory = self.preferred
        visited = set()

        while trajectory.final_state is None:
            if trajectory.key in visited:
                raise RuntimeError(
                    "Circular trajectory state source."
                )

            visited.add(trajectory.key)

            source_key = trajectory.state_source_key

            if source_key not in self.trajectories:
                raise KeyError(
                    f"State source trajectory "
                    f"'{source_key}' does not exist."
                )

            trajectory = self.trajectories[
                source_key
            ]

        return trajectory.final_state

    def _validate_state_sources(self) -> None:
        for trajectory in self.trajectories.values():
            source_key = trajectory.state_source_key

            if (
                source_key is not None
                and source_key
                not in self.trajectories
            ):
                raise ValueError(
                    f"Trajectory '{trajectory.key}' "
                    f"references unknown state source "
                    f"'{source_key}'."
                )
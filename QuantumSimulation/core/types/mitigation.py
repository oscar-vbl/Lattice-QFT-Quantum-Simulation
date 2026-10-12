# QuantumSimulation/core/types/mitigation.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MitigationStepConfig:
    """
    Configuration for a single mitigation strategy.

    Parameters
    ----------
    method:
        Registered mitigation method name, such as
        ``"Richardson"``.
    options:
        Method-specific configuration.
    """

    method: str
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_method = self.method.strip()

        if not normalized_method:
            raise ValueError(
                "Mitigation method cannot be empty."
            )

        object.__setattr__(
            self,
            "method",
            normalized_method,
        )


@dataclass(frozen=True)
class MitigationConfig:
    """
    Configuration for the post-execution mitigation pipeline.

    Mitigation operates on measured trajectories. It does not
    generate quantum states and does not perform time evolution.

    Parameters
    ----------
    enabled:
        Whether the mitigation pipeline is active.
    steps:
        Ordered mitigation strategies.
    preferred_output:
        Optional key of the trajectory that should be selected
        after mitigation.
    """

    enabled: bool = False
    steps: tuple[MitigationStepConfig, ...] = ()
    preferred_output: str | None = None

    def __post_init__(self) -> None:
        if not self.enabled and self.steps:
            raise ValueError(
                "Mitigation steps were provided while mitigation "
                "is disabled."
            )

        if self.enabled and not self.steps:
            raise ValueError(
                "Mitigation is enabled but no mitigation steps "
                "were configured."
            )

        if self.preferred_output is not None:
            normalized_output = (
                self.preferred_output.strip()
            )

            if not normalized_output:
                raise ValueError(
                    "preferred_output cannot be empty."
                )

            object.__setattr__(
                self,
                "preferred_output",
                normalized_output,
            )